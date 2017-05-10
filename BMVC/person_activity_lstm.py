from numpy import genfromtxt
import numpy as np
from keras.utils import to_categorical
from keras.layers import Input, Masking, LSTM, Dropout, Dense
from keras.layers.merge import multiply, average, concatenate
from keras.models import Model, load_model
from keras.optimizers import adam
from matplotlib import pyplot as plt
import h5py
from numpy import savetxt
import keras

data_dir = './ActivityDataset/individual_data'
data_path = './person_actions_posenet_meta_split4.h5'
model_path = './split4/models/person-lstm3-split4.h5'

batch_size = 512
learning_rate = 0.001

dataset = h5py.File(data_path, 'r')
trainX = dataset['trainX'][()].transpose()
trainY = dataset['trainY'][()].transpose() - 1
testX = dataset['testX'][()].transpose()
testY = dataset['testY'][()].transpose() - 1
trainMeta = dataset['trainMeta'][()].transpose()
testMeta = dataset['testMeta'][()].transpose()


trainY[trainY == -1] = 0
trainX = np.reshape(trainX, [-1, 10, 2056])
trainX_pos = trainX[:, :, 0:8]
trainX_img = trainX[:, :, 8:]
trainY = np.reshape(to_categorical(np.reshape(trainY, (-1,)), num_classes=2), (-1, 10, 2))
print(trainY.shape, trainX_img.shape)

testY[testY == -1] = 0
testX = np.reshape(testX, [-1, 10, 2056])
testX_pos = testX[:, :, 0:8]
testX_img = testX[:, :, 8:]
testY = np.reshape(to_categorical(np.reshape(testY, (-1,)), num_classes=2), (-1, 10, 2))
print(testY.shape, testX_img.shape)

# person_lstm = load_model('./split4/models/person-lstm3-split4.h5')
# keras.backend.set_value(person_lstm.optimizer.lr, 0.0001*2)

pos_in = Input(shape=(10, 8))
img_in = Input(shape=(10, 2048))

pos_masked = Masking()(pos_in)
img_masked = Masking()(img_in)

lstm_pos = LSTM(512, activation='sigmoid', recurrent_activation='tanh', return_sequences=True)(pos_masked)
lstm_img = LSTM(256, activation='sigmoid', recurrent_activation='tanh', return_sequences=True)(img_masked)

drop_img = Dropout(0.95)(lstm_img)
drop_pos = Dropout(0.01)(lstm_pos)

fc_pos = (Dense(128, activation='tanh')(drop_pos))
fc_img = (Dense(32, activation='tanh')(drop_img))
merged = concatenate(inputs=[drop_pos, drop_img], axis=2)

# pred_pos = Dense(2, activation='softmax')(fc_pos)
# pred_img = Dense(2, activation='softmax')(fc_img)
# merged = average(inputs=[drop_pos, drop_img])

fc = Dense(2, activation='softmax')(merged)

person_lstm = Model(inputs=[pos_in, img_in], outputs=fc)
print(person_lstm.summary())
optm = adam(lr=learning_rate)
person_lstm.compile(optimizer=optm, loss='binary_crossentropy', metrics=['accuracy'])


arr = []
max_acc = 0.0
for i in range(100):
    person_lstm.fit([trainX_pos, trainX_img], trainY, batch_size=batch_size)
    scores = person_lstm.evaluate([testX_pos, testX_img], testY, batch_size=4096, verbose=0)
    arr.append(scores[1])
    print(scores)
    plt.plot(arr)
    plt.pause(0.001)
    plt.savefig('only-posenet.jpg')
    # if i % 10 == 0 and i is not 0:
    #     batch_size *= 2
    if scores[1] > max_acc and i>30:
        max_acc = scores[1]
        person_lstm.save(model_path)
        Y = person_lstm.predict_on_batch([testX_pos, testX_img])
        savetxt('./split4/atomic/actions.txt', np.reshape(Y, (-1, 20)), delimiter=',')
        savetxt('./split4/atomic/meta.txt', testMeta, delimiter=',')

'''
for seq in range(1, 45):
    trainX = read_csv(data_dir + '/' + 'trainX_%2.2d' % seq, delimiter=',').as_matrix()
    trainY = read_csv(data_dir + '/' + 'trainY_%2.2d' % seq, delimiter=',').as_matrix() - 1
    trainY[trainY == -1] = 0
    trainX = np.reshape(trainX, [-1, 10, 2052])
    trainX_pos = trainX[:, :, 0:4]
    trainX_img = trainX[:, :, 4:]
    trainY = np.reshape(to_categorical(np.reshape(trainY, (-1,)), num_classes=2), (-1, 10, 2))
    print(trainY.shape, trainX_img.shape)
    break
'''

'''
arr = []
for i in range(11000000):
    eval_scores = []
    eval_num = []
    for seq in range(1, 45):
        if seq not in test_seq:
            trainX = read_csv(data_dir + '/' + 'trainX_%2.2d' % seq, delimiter=',').as_matrix()
            trainY = read_csv(data_dir + '/' + 'trainY_%2.2d' % seq, delimiter=',').as_matrix() - 1
            trainY[trainY == -1] = 0
            trainX = np.reshape(trainX, [-1, 10, 2052])
            trainX_pos = trainX[:, :, 0:4]
            trainX_img = trainX[:, :, 4:]
            trainY = np.reshape(to_categorical(np.reshape(trainY, (-1,)), num_classes=2), (-1, 10, 2))
            person_lstm.fit([trainX_pos, trainX_img], trainY, batch_size=batch_size)

        if seq in test_seq:
            trainX = read_csv(data_dir + '/' + 'trainX_%2.2d' % seq, delimiter=',').as_matrix()
            trainY = read_csv(data_dir + '/' + 'trainY_%2.2d' % seq, delimiter=',').as_matrix() - 1
            trainY[trainY == -1] = 0
            trainX = np.reshape(trainX, [-1, 10, 2052])
            trainX_pos = trainX[:, :, 0:4]
            trainX_img = trainX[:, :, 4:]
            trainY = np.reshape(to_categorical(np.reshape(trainY, (-1,)), num_classes=2), (-1, 10, 2))
            scores = person_lstm.evaluate([trainX_pos, trainX_img], trainY, batch_size=batch_size)
            eval_scores.append(scores)
            eval_num.append(trainY.shape[0])
    arr.append(np.sum(np.array(eval_scores) * np.array(eval_num)) / np.sum(np.array(eval_num)))
    plt.plot(arr)
    plt.pause(0.001)
'''