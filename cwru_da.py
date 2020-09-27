# -- coding: utf-8 --
import utils
import keras
import keras.layers as KL
import keras.backend as K
from keras.models import Model
import numpy as np
import pickle
import scipy.io as sio
import time
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score
from collections import OrderedDict
import keras.backend.tensorflow_backend as KTF
import h5py



class GAN_1D():
    def __init__(self):
        self.len_segment = 4096
        self.len_data = 1
        self.time_shift = 1
        self.clip_value = 0.01

    def wasserstein_loss(self, y_true, y_pred):
        return -K.mean(y_true * y_pred)

    def confidence_loss(self, y_true, y_pred):
        return K.sum(K.square(K.max(y_true, axis=0) - K.max(y_pred, axis=0)))


    def build_M(self):
        inp = KL.Input(shape=(self.len_data, self.len_segment, 1))
        x = inp

        x = KL.Reshape((self.len_segment,1))(x)
        x = KL.Conv1D(128,17,strides=1,padding='same',name='conv1')(x)
        x = KL.BatchNormalization(name='bn1')(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.MaxPool1D(16, name='pool1')(x)
        x = KL.Conv1D(128,17,strides=1,padding='same',name='conv2')(x)
        x = KL.BatchNormalization(name='bn2')(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.MaxPool1D(16, name='pool2')(x)
        x = KL.Conv1D(128,3,padding='same',name='conv3')(x)
        x = KL.BatchNormalization(name='bn3')(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.MaxPool1D(2, name='pool3')(x)

        x = KL.Flatten()(x)
        out = x
        # out_deconv = y
        return Model([inp], [out])

    def build_C(self):
        inp = KL.Input(shape=(1024,))
        x = inp
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(256)(x)
        x = KL.BatchNormalization()(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(4, activation='softmax')(x)
        out = x
        return Model(inp, out)

    def build_D(self):
        inp = KL.Input(shape=(1024,))
        x = inp
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(512)(x)
        x = KL.BatchNormalization()(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(128)(x)
        x = KL.BatchNormalization()(x)
        x = KL.LeakyReLU(0.2)(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(2, activation='linear')(x)
        out = x
        return Model(inp, out)


    def train_s2(self, epochs, epoch_size, batch_size=16, save_interval=10):
        # load dataset
        file1_va = h5py.File('./data_da/data1772.h5', 'r')
        train_data = file1_va['data1772_x'][:]
        train_data = train_data.reshape(len(train_data), 1, 4096, 1)
        train_label = file1_va['data1772_y'][:]
        file2_va = h5py.File('./data_da/data1750.h5', 'r')
        test_data = file2_va['data1750_x'][:]
        test_data = test_data.reshape(len(test_data), 1, 4096, 1)
        test_label = file2_va['data1750_y'][:]

        train_data = utils.normalize_data(train_data, 'std')
        test_data = utils.normalize_data(test_data, 'std')

####################################################
        index1 = np.arange(np.size(train_data, 0))
        np.random.shuffle(index1)

        train_data_all = train_data[index1, :, :, :]
        train_label_all = train_label[index1, :]

        train_data = train_data_all[:, :, :, :]
        train_label = train_label_all[:, :]
###################################################
        index = np.arange(np.size(test_label, 0))
        np.random.shuffle(index)

        test_data_all = test_data[index, :, :, :]
        test_label_tru = test_label[index, :]
###################################################
        # 测试数据集伪标签
        index2 = np.arange(np.size(test_label, 0))
        np.random.shuffle(index2)

        data = sio.loadmat('./data_da/gan_fea_pca4test_AB.mat')
        test_pre = data['test_pre']
        idx = np.argmax(test_pre, axis=1)
        test_label = keras.utils.to_categorical(idx)
        test_label = test_label[index2, :]
        test_data = test_data[index2, :, :, :]


        # source_data.name = 'cwru_data_12k'

        ms = self.build_M()
        # ms.load_weights('./net_weights/MS10.hdf5')
        ms.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        ms.summary()
        c = self.build_C()
        # c.load_weights('./net_weights/C10.hdf5')
        c.compile(optimizer=keras.optimizers.Adam(5e-4), loss='categorical_crossentropy', metrics=['acc'])
        d = self.build_D()
        # d.load_weights('./net_weights/best_v.hdf5')
        # d.compile(optimizer=keras.optimizers.Adam(5e-4), loss='mse', metrics=['acc'])
        d.compile(optimizer=keras.optimizers.Adam(5e-4), loss='mse', metrics=['acc'])

        inp = KL.Input(shape=(self.len_data, self.len_segment, 1))
        fea = ms(inp)
        valid = d(fea)  # 判别训练数据和测试数据的特征相似度
        classify = c(fea)  # 分类器
        # for l in ms.layers:
        #     if l.name in ['conv1','conv2','conv3']:
        #         l.trainable = False
        d.trainable = False
        mt_p_c_d = Model(inp, [classify, valid])
        mt_p_c_d.compile(optimizer=keras.optimizers.Adam(5e-4), loss=['categorical_crossentropy', 'mse'],
                         loss_weights=[1, 1], metrics=['acc'])
        mt_p_c_d.summary()
        self.c_m_acc = 0
        record = OrderedDict({'Dloss': [], 'Dacc': [], 'Gloss': [], 'Gacc': [], 'Cacc': []})
        epoch_record = OrderedDict({'Closs': [], 'Cacc': [], 'Sloss': [], 'Sacc': []})
        for i in range(epochs):

            for j in range(int(epoch_size / batch_size / 8)):

                for k in range(2):
                    temp_idx = np.random.randint(0, epoch_size, batch_size)
                    d_fea = ms.predict(np.concatenate([train_data[temp_idx,], test_data[temp_idx,]], axis=0))
                    d_l = keras.utils.to_categorical(
                        np.array(([1] * batch_size + [0] * batch_size)))  # 给训练数据和测试数据的特征加标签，训练集标签为1,测试集标签为0
                    d_loss = d.train_on_batch(d_fea, d_l)  # 判别数据真假

                sample_weights = [np.array(([1] * batch_size + [0.7] * batch_size)), np.ones((batch_size * 2,))]
                # sample_weigh---主要解决的是样本质量不同的问题，比如前1000个样本的可信度，那么它的权重就要高，后1000个样本可能有错、不可信，那么权重就要调低。
                # retrain时, 测试数据的预测标签参与训练，第二个batch_size前的参数可以取[0.1, 1.0]之间的值。
                for k in range(1):
                    temp_idx = np.random.randint(0, epoch_size, batch_size)
                    g_data = np.concatenate([train_data[temp_idx,], test_data[temp_idx,]], axis=0)
                    g_label = np.concatenate([train_label[temp_idx,], test_label[temp_idx,]], axis=0)  # 训练和测试数据的真实标签，retrain时用测试数据的伪标签
                    g_valid = keras.utils.to_categorical(
                        np.array(([0] * batch_size + [1] * batch_size)))  # 区分训练和测试的真假的标签
                    # g_valid = np.array(([-1]*batch_size+[1]*batch_size))
                    g_loss = mt_p_c_d.train_on_batch(g_data, [g_label, g_valid],
                                                     sample_weight=sample_weights)  # 训练目的：训练数据和测试数据特征距离最小，分类正确率最高

                '''if j % 20 == 0:
                    print (
                                "%d epoch %d batch [D loss: %f, acc.: %.2f%%] [G loss: %f, acc.: %.2f%%] [classify acc.: %.2f%%]" % (
                        i, j, d_loss[0], 100 * d_loss[1], g_loss[2], 100 * g_loss[4], 100 * g_loss[3]))'''

            train_fea = ms.predict(train_data_all)
            train_eva = c.evaluate(train_fea, train_label_all)
            print(train_eva)
            test_fea = ms.predict(test_data_all)
            test_eva = c.evaluate(test_fea, test_label_tru)
            test_r = c.predict(test_fea)
            ypred = np.argmax(test_r, axis=1)
            yTrue = np.argmax(test_label_tru, axis=1)
            dd = np.argwhere(ypred==yTrue)
            dd = np.array(dd)
            acc = float(len(dd))/float(test_label_tru.shape[0])


            print(test_eva)
            epoch_record['Closs'].append(round(test_eva[0], 6))
            epoch_record['Cacc'].append(round(test_eva[1], 6))
            epoch_record['Sloss'].append(round(train_eva[0], 6))
            epoch_record['Sacc'].append(round(train_eva[1], 6))
            utils.save_dict(epoch_record, 'epoch_record')
            if test_eva[1] >= self.c_m_acc:
                self.c_m_acc = test_eva[1]
                ms.save_weights('./net_weights/best_mt.hdf5')
                d.save_weights('./net_weights/best_v.hdf5')
                c.save_weights('./net_weights/best_c.hdf5')
            print('optimal_acc',self.c_m_acc,'pred_acc: ', acc)

            if (i + 1) % 50 == 0:
                K.set_value(mt_p_c_d.optimizer.lr, K.get_value(mt_p_c_d.optimizer.lr) * 0.5)
                K.set_value(d.optimizer.lr, K.get_value(d.optimizer.lr) * 0.5)
                # mt_p_c_d.compile(optimizer=keras.optimizers.Adam(5e-*0.9**(i/5)),loss=['categorical_crossentropy','categorical_crossentropy'],loss_weights=[1,1],metrics=['acc'])
                # d.compile(optimizer=keras.optimizers.Adam(5e-*0.9**(i/7)),loss='categorical_crossentropy',metrics=['acc'])

        ms.save_weights('./net_weights/last_mt.hdf5')
        d.save_weights('./net_weights/last_v.hdf5')
        c.save_weights('./net_weights/last_c.hdf5')

    def test(self):
        # load dataset
        file1_va = h5py.File('./data_da/data1750.h5', 'r')
        train_data = file1_va['data1750_x'][:]
        train_data = train_data.reshape(len(train_data), 1, 4096, 1)
        train_label = file1_va['data1750_y'][:]
        file2_va = h5py.File('./data_da/data1730.h5', 'r')
        test_data = file2_va['data1730_x'][:]
        test_data = test_data.reshape(len(test_data), 1, 4096, 1)
        test_label = file2_va['data1730_y'][:]

        train_data = utils.normalize_data(train_data, 'std')
        test_data = utils.normalize_data(test_data, 'std')

        ms = self.build_M()
        ms.load_weights('./net_weights/best_mt.hdf5')
        ms.compile(optimizer=keras.optimizers.Adam(), loss='mse')
        c = self.build_C()
        c.load_weights('./net_weights/best_c.hdf5')
        c.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy')

        train_fea = ms.predict(train_data)
        test_fea = ms.predict(test_data)
        test_pre = c.predict(test_fea)
        sio.savemat('gan_fea_pca4test_BA_7.mat',
                    {'train_fea': train_fea, 'train_label': train_label, 'test_fea': test_fea, 'test_label': test_label,
                     'test_pre': test_pre})

        '''layer_name = 'conv1'
        intermediate_layer_model = Model(inputs=ms.input, outputs=ms.get_layer(layer_name).output)
        tt_conv1 = intermediate_layer_model.predict(test_data)
        sio.savemat('tt_conv1.mat', {"sonar": tt_conv1})

        layer_name = 'conv2'
        intermediate_layer_model = Model(inputs=ms.input, outputs=ms.get_layer(layer_name).output)
        tt_conv2 = intermediate_layer_model.predict(test_data)
        sio.savemat('tt_conv2.mat', {"sonar": tt_conv2})

        layer_name = 'pool1'
        intermediate_layer_model = Model(inputs=ms.input, outputs=ms.get_layer(layer_name).output)
        tt_pool1 = intermediate_layer_model.predict(test_data)
        sio.savemat('tt_pool1.mat', {"sonar": tt_pool1})'''


    def save_net(self, net, name):
        time_str = time.strftime('%H_%M_%S', time.localtime(time.time()))
        net.save_weights('./net_weights/' + name + '_' + time_str + '.hdf5')


if __name__ == '__main__':
    acgan = GAN_1D()
    #acgan.train_s1(50, batch_size=16)
    acgan.train_s2(120, 60000, batch_size=16, save_interval=10)
    #acgan.test()
    #print(acgan.c_m_acc)