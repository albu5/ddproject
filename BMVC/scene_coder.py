from numpy import genfromtxt
import numpy as np
from keras.layers import Input, Dense


X = []
for i in range(1, 45):
    seq_dir = './ActivityDataset' + '/seq%2.2d' % i
    print(seq_dir)
    file_path = seq_dir + '/resnet50.txt'
    resnet_out = genfromtxt(file_path, delimiter=',')
    resnet_out = resnet_out[11:10:, :].tolist()
    for vec in resnet_out:
        X.append(vec)

trainX = np.array(X)
print(trainX.shape)

