from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt
import random
from keras import backend as KB

vec_len = 100
num_iter = 10000


def get_batch(bs=512, noise=0.1):
    x = np.zeros(shape=(bs, vec_len))
    y = np.zeros(shape=(bs, 2))
    for i in range(bs):
        toss = bool(random.getrandbits(1))
        slope = np.random.uniform()
        linspace = np.array(list(range(vec_len)))
        if toss:
            x[i][:] = np.sin(slope*linspace) + noise*np.random.uniform(size=vec_len)
            y[i][:] = np.array([0, 1])
        else:
            x[i][:] = (slope*linspace) + noise*np.random.uniform(size=vec_len)
            y[i][:] = np.array([1, 0])
    return np.expand_dims(x, axis=2), y


# X, Y = get_batch(1, 1)
# plt.plot(np.squeeze(X))
# plt.show()
# plt.pause(0.5)
# plt.close()
# print(Y)

model = Sequential()
model.add(LSTM(16, input_shape=(vec_len, 1), return_sequences=True, activation='sigmoid'))
model.add(LSTM(8, return_sequences=False, activation='sigmoid'))
model.add(Dense(2, activation=KB.softmax))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

for i in range(num_iter):
    X, Y = get_batch(200, 1)
    model.train_on_batch(X, Y)
    score, acc = model.evaluate(X, Y, batch_size=200, verbose=1)
    print('Score: ', score, 'Accuracy:', acc)