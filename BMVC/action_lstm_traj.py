from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Masking
import numpy as np
import matplotlib.pyplot as plt
import random
from keras import backend as kb
from keras.optimizers import adam
from utils import get_batch


seq_len = 32
batch_size = 1024*8
num_iter = 10000000
decay_step = 100
disp_step = 1

plt.ion()

model = Sequential()
model.add(Masking(input_shape=(seq_len, 4), mask_value=0))
# model.add(SimpleRNN(128, return_sequences=True, activation='sigmoid', use_bias=True))
model.add(LSTM(256, return_sequences=True,
               activation='sigmoid', recurrent_activation='tanh',
               use_bias=True, unit_forget_bias=True))
model.add(LSTM(64, return_sequences=True,
               activation='sigmoid', recurrent_activation='tanh',
               use_bias=True, unit_forget_bias=True))

# model.add(LSTM(32, return_sequences=True, activation='sigmoid', use_bias=True))
# model.add(Dense(8, activation=kb.sigmoid))
model.add(Dense(16, activation='tanh'))
model.add(Dense(2, activation=kb.softmax))

optimizer = adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

data_dir = './ActivityDataset'
traj_dir = data_dir + '/' + 'TrajectoriesLong'
train_dir = traj_dir + '/' + 'train'


acc_arr = []
for i in range(num_iter):
    X, Y = get_batch(train_dir, batch_size, seq_len)
    model.train_on_batch(X, Y)
    score, acc = model.evaluate(X, Y, batch_size=batch_size, verbose=1)
    acc_arr.append(acc)
    print('Step:', i, 'Score: ', score, 'Accuracy:', acc)
    if (i % decay_step == 0) and i is not 0:
        kb.set_value(optimizer.lr, 0.5 * kb.get_value(optimizer.lr))
    if (i % disp_step == 0) and i is not 0:
        plt.plot(acc_arr)
        plt.pause(0.0005)
