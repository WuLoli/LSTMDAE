import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']= '0'
import numpy as np
import pickle
import tensorflow as tf
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import init_notebook_mode, iplot
import time
import seaborn as sn
import pandas as pd
import h5py
init_notebook_mode()

# parameters
data_path = '../../../data/RML2018.01a/'
file_name = 'NormalClasses.hdf5'

signal_len = 1024
modulation_num = 11

def get_amp_phase(data):
    X_train_cmplx = data[:, 0, :] + 1j * data[:, 1, :]
    X_train_amp = np.abs(X_train_cmplx)
    X_train_ang = np.arctan2(data[:, 1, :], data[:, 0, :]) / np.pi
    X_train_amp = np.reshape(X_train_amp, (-1, 1, signal_len))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, signal_len))
    X_train = np.concatenate((X_train_amp, X_train_ang), axis = 1) 
    X_train = np.transpose(np.array(X_train), (0, 2, 1))
    for i in range(X_train.shape[0]):
        X_train[i, :, 0] = X_train[i, :, 0] / np.linalg.norm(X_train[i, :, 0], 2)
    
    return X_train

def set_up_data(data_path, file_name):
    f = h5py.File(data_path + file_name, 'r')
    data = f['X'][()]
    label = f['Y'][()]
    SNR = f['Z'][()]
    
    classes = ['32PSK',
             '16APSK',
             '32QAM',
             'FM',
             'GMSK',
             '32APSK',
             'OQPSK',
             '8ASK',
             'BPSK',
             '8PSK',
             'AM-SSB-SC',
             '4ASK',
             '16PSK',
             '64APSK',
             '128QAM',
             '128APSK',
             'AM-DSB-SC',
             'AM-SSB-WC',
             '64QAM',
             'QPSK',
             '256QAM',
             'AM-DSB-WC',
             'OOK',
             '16QAM']
    
    dic = {}
    for i in np.unique(label):
        dic[int(i)] = classes[int(i)]

    modulation_index = {'FM': 0, 'GMSK': 1, 'OQPSK': 2, 'BPSK': 3, '8PSK': 4, 'AM-SSB-SC': 5, '4ASK': 6, 'AM-DSB-SC': 7, 'QPSK': 8, 'OOK': 9, '16QAM': 10}

    label = np.reshape(label, (-1, 1))
    for i in range(label.shape[0]):
        label[i, 0] = modulation_index[dic[label[i, 0]]]

    label = np.concatenate((label, SNR), axis = 1)
    
    index = list(range(data.shape[0]))
    np.random.seed(2019)
    np.random.shuffle(index)

    train_proportion = 0.5
    validation_proportion = 0.25
    test_proportion = 0.25

    X_train = data[index[:int(data.shape[0] * train_proportion)], :, :]
    Y_train = label[index[:int(data.shape[0] * train_proportion)], :]
    X_validation = data[index[int(data.shape[0] * train_proportion) : int(data.shape[0] * (train_proportion + validation_proportion))], :, :]
    Y_validation = label[index[int(data.shape[0] * train_proportion) : int(data.shape[0] * (train_proportion + validation_proportion))], :]
    X_test = data[index[int(data.shape[0] * (train_proportion + validation_proportion)):], :, :]
    Y_test = label[index[int(data.shape[0] * (train_proportion + validation_proportion)):], :]
    
    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test, modulation_index

def zero_mask(X_train, p):
    num = int(X_train.shape[1] * p)
    res = X_train.copy()
    index = np.array([[i for i in range(X_train.shape[1])] for _ in range(X_train.shape[0])])
    for i in range(index.shape[0]):
        np.random.shuffle(index[i, :])
    
    for i in range(res.shape[0]):
        res[i, index[i, :num], :] = 0
        
    return res

# set up data
X_train, Y_train, X_validation, Y_validation, X_test, Y_test, modulation_index = set_up_data(data_path, file_name)

X_train = np.moveaxis(X_train, 1, 2)
X_validation = np.moveaxis(X_validation, 1, 2)
X_test = np.moveaxis(X_test, 1, 2)

X_train = get_amp_phase(X_train)
X_validation = get_amp_phase(X_validation)
X_test = get_amp_phase(X_test)

Y_train = Y_train.astype(int)
Y_validation = Y_validation.astype(int)
Y_test = Y_test.astype(int)

encoder_inputs = tf.keras.Input(shape = (X_train.shape[1], X_train.shape[2]),
                                name = 'encoder_inputs')

encoder_1, state_h_1, state_c_1 = tf.keras.layers.CuDNNLSTM(units = 32,
                                    return_sequences = True,
                                    return_state = True,
                                    name = 'encoder_1')(encoder_inputs)

drop_prob = 0.2
drop_1 = tf.keras.layers.Dropout(drop_prob, name = 'drop_1')(encoder_1)

encoder_2, state_h_2, state_c_2 = tf.keras.layers.CuDNNLSTM(units = 32,
                                    return_state = True,
                                    return_sequences = True,                
                                    name = 'encoder_2')(drop_1)

decoder = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2),
                                          name = 'decoder')(encoder_2)

# 3 Dense layers for classification with bn
clf_dropout = 0.2

clf_dense_1 = tf.keras.layers.Dense(units = 32,
                                    activation = tf.nn.relu,
                                    name = 'clf_dense_1')(state_h_2)

bn_1 = tf.keras.layers.BatchNormalization(name = 'bn_1')(clf_dense_1)

clf_drop_1 = tf.keras.layers.Dropout(clf_dropout, name = 'clf_drop_1')(bn_1)

clf_dense_2 = tf.keras.layers.Dense(units = 16,
                                    activation = tf.nn.relu,
                                    name = 'clf_dense_2')(clf_drop_1)

bn_2 = tf.keras.layers.BatchNormalization(name = 'bn_2')(clf_dense_2)

clf_drop_2 = tf.keras.layers.Dropout(clf_dropout, name = 'clf_drop_2')(bn_2)

clf_dense_3 = tf.keras.layers.Dense(units = modulation_num,
                                    name = 'clf_dense_3')(clf_drop_2)

softmax = tf.keras.layers.Softmax(name = 'softmax')(clf_dense_3)

model = tf.keras.Model(inputs = encoder_inputs, outputs = [decoder, softmax])
model.summary()


learning_rate = 10 ** -3
lam = 0.1

model.compile(loss = ['mean_squared_error', 'categorical_crossentropy'],
              loss_weights = [1 - lam, lam],
              metrics=['accuracy'],
              optimizer = tf.keras.optimizers.Adam(lr = learning_rate))

best = 0
train_acc = []
val_acc = []


for ite in range(150):
    X_train_masked = zero_mask(X_train, 0.1)
    print(ite)
    history = model.fit(x = X_train,
                        y = [X_train, tf.keras.utils.to_categorical(Y_train[:, 0])],
                        validation_data = (X_validation, [X_validation, tf.keras.utils.to_categorical(Y_validation[:, 0])]),
                        batch_size = 128,
                        epochs = 1)
    
    train_acc.append(history.history['softmax_acc'][0])
    val_acc.append(history.history['val_softmax_acc'][0])

    if history.history['val_softmax_acc'][0] > best:
        best = history.history['val_softmax_acc'][0]
        model.save('DAELSTM.h5')

    with open('val_result.txt', 'a') as f:
        f.write(str(history.history['val_softmax_acc'][0] * 100) + '\n')
        

clf = tf.keras.models.load_model('DAELSTM.h5')

res = clf.predict(X_test)[1]
res = np.argmax(res, axis = 1)
test_accuracy = {}
for i in range(X_test.shape[0]):
    if Y_test[i, 1] not in test_accuracy:
        if Y_test[i, 0] == res[i]:
            test_accuracy[Y_test[i, 1]] = [1, 1]
        else:
            test_accuracy[Y_test[i, 1]] = [0, 1]
    else:
        if Y_test[i, 0] == res[i]:
            test_accuracy[Y_test[i, 1]][0] += 1
            test_accuracy[Y_test[i, 1]][1] += 1
        else:
            test_accuracy[Y_test[i, 1]][1] += 1

nomi = 0
deno = 0
for snr in test_accuracy:
    nomi += test_accuracy[snr][0]
    deno += test_accuracy[snr][1]

best = nomi / deno

with open('result.txt', 'a') as f:
    for item in [test_accuracy[i][0] / test_accuracy[i][1] for i in np.sort(list(test_accuracy))]:
        f.write(str(item * 100) + '\n')
        
    f.write(str(best * 100) + '\n')
    f.write('\n')