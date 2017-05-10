from keras.layers import Input, LSTM, Dense, Masking, Dropout
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.optimizers import adam
from keras.utils import to_categorical
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import keras


data_dir = './ind2scene/'
model_path = data_dir + '/' + 'jump_net.h5'
learning_rate = 0.001

trainX1 = np.reshape(genfromtxt(data_dir + '/' + 'trainX1.csv', delimiter=','), newshape=(-1, 10, 20*11))
trainX2 = np.reshape(genfromtxt(data_dir + '/' + 'trainX2.csv', delimiter=','), newshape=(-1, 10, 2048))
trainY = to_categorical(genfromtxt(data_dir + '/' + 'trainY.csv', delimiter=',') - 1)
testX1 = np.reshape(genfromtxt(data_dir + '/' + 'testX1.csv', delimiter=','), newshape=(-1, 10, 20*11))
testX2 = np.reshape(genfromtxt(data_dir + '/' + 'testX2.csv', delimiter=','), newshape=(-1, 10, 2048))
testY = to_categorical(genfromtxt(data_dir + '/' + 'testY.csv', delimiter=',') - 1)

print(trainX1.shape, trainX1.shape, trainY.shape, testX1.shape, testX2.shape, testY.shape)


pos_in = Input(shape=(10, 20*11))
img_in = Input(shape=(10, 2048))

pos_masked = Masking()(pos_in)
img_masked = Masking()(img_in)

lstm_pos = LSTM(256, activation='sigmoid', recurrent_activation='tanh', return_sequences=False)(pos_masked)
lstm_img = LSTM(256, activation='sigmoid', recurrent_activation='tanh', return_sequences=False)(img_masked)

drop_img = Dropout(0.98)(lstm_img)
drop_pos = Dropout(0.5)(lstm_pos)

fc_pos = (Dense(32, activation='tanh')(drop_img))
fc_img = (Dense(16, activation='tanh')(drop_pos))
merged = concatenate(inputs=[fc_pos, fc_img], axis=1)
#
# pred_pos = Dense(2, activation='softmax')(fc_pos)
# pred_img = Dense(2, activation='softmax')(fc_img)
# merged = average(inputs=[drop_pos, drop_img])

fc = Dense(5, activation='softmax')(merged)

jump_net = Model(inputs=[pos_in, img_in], outputs=fc)
print(jump_net.summary())
optm = adam(lr=learning_rate)
jump_net.compile(optimizer=optm, loss='binary_crossentropy', metrics=['accuracy'])


arr = []
max_acc = 0.0
for i in range(1000000):
    jump_net.fit(x=[trainX1, trainX2], y=trainY, batch_size=512, epochs=1, verbose=1)
    scores = jump_net.evaluate([testX1, testX2], testY, batch_size=2048, verbose=0)
    arr.append(scores[1])
    plt.plot(arr)
    plt.pause(0.01)
    plt.savefig('temp.jpg')
    if scores[1] > max_acc and i > 30:
        max_acc = scores[1]
        jump_net.save(model_path)
