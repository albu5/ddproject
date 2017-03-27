import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import adam, rmsprop
from keras import backend as kb
from keras.layers import regularizers
import matplotlib.pyplot as plt

plt.ion()
disp_step = 10
decay_step = 500
num_iter = 100000
slope_dir = './ActivityDataset/trajSlopes'

train_data = np.genfromtxt(slope_dir + '/' + 'train_A.txt', delimiter=',')
train_x = train_data[:, 0:12].astype(np.float32)
train_y = train_data[:, 12]
train_y = np.transpose(np.vstack([train_y == 1, train_y == 2])).astype(np.float32)
print(train_y.shape, train_x.shape)
print(train_x)
print(train_y)

test_data = np.genfromtxt(slope_dir + '/' + 'test_A.txt', delimiter=',')
test_x = test_data[:, 0:12].astype(np.float32)
test_y = test_data[:, 12]
test_y = np.transpose(np.vstack([test_y == 1, test_y == 2])).astype(np.float32)


model = Sequential()
model.add(Dense(24, input_shape=(12,), activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(12, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(12, activation='sigmoid', use_bias=True, kernel_regularizer=regularizers.l2(0.001)))
model.add(Dense(2, activation='softmax', use_bias=True, kernel_regularizer=regularizers.l2(0.001)))

optimizer = adam(lr=0.01/2)

model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
acc_arr = []
for i in range(num_iter):
    model.train_on_batch(train_x, train_y)
    score, acc = model.evaluate(train_x, train_y, verbose=0)
    acc_arr.append(acc)
    print('Step:', i, 'Score: ', score, 'Accuracy:', acc)
    if (i % decay_step == 0) and i is not 0:
        kb.set_value(optimizer.lr, 0.5 * kb.get_value(optimizer.lr))
    if (i % disp_step == 0) and i is not 0:
        plt.plot(acc_arr)
        plt.pause(0.0005)
