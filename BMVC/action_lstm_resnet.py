from keras.models import Model
from keras.layers import Dense, Input, TimeDistributed, Reshape
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Masking
import numpy as np
import matplotlib.pyplot as plt
import random
from keras import backend as kb
from keras.optimizers import adam
from utils import get_batch, get_immediate_subdirectories, get_batch_resnet, get_batch_image
from keras.applications.resnet50 import ResNet50
from keras.layers.core import K


K.set_learning_phase(0)


seq_len = 9
batch_size = 2
num_iter = 10000000
decay_step = 100
disp_step = 1

plt.ion()

resnet = ResNet50(weights='imagenet', include_top=False)
# print(resnet.summary())

in_layer = Input(shape=(seq_len, 224*224*3))
im_layer = Reshape(target_shape=(seq_len, 224, 224, 3))(in_layer)
out_layer = TimeDistributed(resnet)(im_layer)
res_squeezed = Reshape(target_shape=(seq_len, 2048))(out_layer)
lstm_out = LSTM(256, return_sequences=True,
                activation='sigmoid', recurrent_activation='tanh',
                use_bias=True, unit_forget_bias=True)(res_squeezed)
prediction_layer = Dense(2, activation=kb.softmax)(lstm_out)

model = Model(inputs=in_layer, outputs=prediction_layer)
optimizer = adam(lr=0.00001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())
data_dir = './ActivityDataset'
traj_dir = data_dir + '/' + 'TrajectoriesLong'
train_dir = traj_dir + '/' + 'train'

ex_dirs = get_immediate_subdirectories(train_dir)

acc_arr = []
for i in range(num_iter):
    X, Y = get_batch_image(train_dir, batch_size, seq_len)
    print(X.shape, Y.shape)
    # model.train_on_batch(X, Y)
    model.fit(x=X, y=Y, epochs=1, verbose=1)
    score, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=1)
    acc_arr.append(acc)
    print('Step:', i, 'Score: ', score, 'Accuracy:', acc)
    if (i % decay_step == 0) and i is not 0:
        kb.set_value(optimizer.lr, 0.5 * kb.get_value(optimizer.lr))
    if (i % disp_step == 0) and i is not 0:
        plt.plot(acc_arr)
        plt.pause(0.0005)
