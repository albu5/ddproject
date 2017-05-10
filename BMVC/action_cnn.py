import numpy as np
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Model
from keras.layers import Dense, Reshape, Dropout
from keras.optimizers import adam
from matplotlib import pyplot as plt
import h5py


data_path = './action_cnn.h5'
batch_size = 32
learning_rate = 0.0001

dataset = h5py.File(data_path, 'r')
trainX = dataset['trainX'][()].transpose()
trainY = dataset['trainY'][()].transpose() - 1
testX = dataset['testX'][()].transpose()
testY = dataset['testY'][()].transpose() - 1

trainX = preprocess_input(np.reshape(trainX, [-1, 224, 224, 3]))
trainY = to_categorical(trainY)
print(trainY.shape, trainX.shape)

testX = preprocess_input(np.reshape(testX, [-1, 224, 224, 3]))
testY = to_categorical(testY)
print(testY.shape, testX.shape)


resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
res_out = resnet.output
res_flat = Reshape(target_shape=(2048, ))(res_out)
fc = Dropout(0.99)(Dense(1024, activation='relu')(res_flat))
pred = Dense(2, activation='softmax')(fc)
action_cnn = Model(inputs=resnet.input, outputs=pred)
print(action_cnn.summary())

optm = adam(lr=learning_rate)
action_cnn.compile(optimizer=optm, loss='binary_crossentropy', metrics=['accuracy'])

arr = []

for i in range(100000):
    action_cnn.fit(trainX, trainY, batch_size=batch_size)
    scores = action_cnn.evaluate(testX, testY, batch_size=batch_size, verbose=0)
    arr.append(scores[1])
    plt.plot(arr)
    plt.pause(0.01)
