from numpy import genfromtxt
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import adam
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import numpy as np

'''
========================================================================================================================
CONSTANTS
========================================================================================================================
'''
batch_size = 32
num_iter = 10000000
decay_step = 10000000
save_step = 1000
disp_step = 10
eval_step = 10
learning_rate = 0.0001
model_path = './models-group-activity-fc.h5'
data_path = './ActivityDataset/group-activity-data-resnet'


'''
========================================================================================================================
MODEL
========================================================================================================================
'''

input_layer = Input(shape=(2228,))
fc1 = Dense(2**3, activation='relu')(input_layer)
fc2 = Dense(2**4, activation='relu')(fc1)
fc3 = Dense(2**2, activation='softmax')(fc1 )
actnet = Model(inputs=input_layer, outputs=fc3)

print(actnet.summary())

optimizer = adam(lr=learning_rate)
actnet.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


'''
========================================================================================================================
TRAINING
========================================================================================================================
'''
trainX = genfromtxt(fname=data_path+'/trainX.txt', delimiter=',')
trainY = to_categorical(genfromtxt(fname=data_path+'/trainY.txt', delimiter=',')-1)
testX = genfromtxt(fname=data_path+'/testX.txt', delimiter=',')
testY = to_categorical(genfromtxt(fname=data_path+'/testY.txt', delimiter=',')-1)

print(trainX.shape, trainY.shape, testX.shape, testY.shape)

score_arr = []
for i in range(num_iter):
    actnet.fit(trainX, trainY, batch_size=128, epochs=1, verbose=1)
    scores = actnet.evaluate(testX, testY, verbose=1)
    score_arr.append(scores[1])
    plt.plot(score_arr)
    plt.pause(0.01)
