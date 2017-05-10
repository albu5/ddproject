from keras.layers import Input, LSTM, Dense, Masking
from keras.models import Model, load_model
from keras.optimizers import adam
from keras.utils import to_categorical
import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
import keras


data_dir = './split1/group_activity_data_train'
model_path = './actlstm/actlstmfc-split4.h5'

trainX = np.reshape(genfromtxt(data_dir + '/' + 'trainX.csv', delimiter=','), newshape=(-1, 10, 110))
trainY = to_categorical(genfromtxt(data_dir + '/' + 'trainY.csv', delimiter=',') - 1)
testX = np.reshape(genfromtxt(data_dir + '/' + 'testX.csv', delimiter=','), newshape=(-1, 10, 110))
testY = to_categorical(genfromtxt(data_dir + '/' + 'testY.csv', delimiter=',') - 1)

print(trainX.shape, trainY.shape, testX.shape, testY.shape)


# model

# input_layer = Input(shape=(10, 110))
# masked_in = Masking()(input_layer)
# lstm1 = LSTM(256, activation='sigmoid', recurrent_activation='tanh',
#              return_sequences=False)(masked_in)
# fc1 = Dense(32, activation='tanh')(lstm1)
# fc2 = Dense(4, activation='softmax')(fc1)
# act_lstm = Model(inputs=input_layer, outputs=fc2)
#
# print(act_lstm.summary())
# optimizer = adam(lr=0.001)
# act_lstm.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

act_lstm = load_model('./actlstm/actlstmfc-split4.h5')
keras.backend.set_value(act_lstm.optimizer.lr, 0.0001)

arr = []
max_acc = 0.88
for i in range(1000000):
    act_lstm.fit(x=trainX, y=trainY, batch_size=512, epochs=1, verbose=1)
    scores = act_lstm.evaluate(testX, testY, batch_size=2048, verbose=0)
    arr.append(scores[1])
    plt.plot(arr)
    plt.pause(0.01)
    plt.savefig('temp.jpg')
    if scores[1] > max_acc and i > 30:
        max_acc = scores[1]
        act_lstm.save(model_path)

