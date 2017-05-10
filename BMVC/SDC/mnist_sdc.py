from keras.datasets import mnist
from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, Input, MaxPooling2D, Flatten
from keras.optimizers import adam
from keras.losses import mean_squared_error as mse
from keras.utils import to_categorical
import numpy as np
from numpy import savetxt
from numpy.linalg import norm, svd
from matplotlib import pyplot as plt

#
# def orthonormalize(a):
#     tall = False
#     if a.shape[0] > a.shape[1]:
#         tall = True
#         a = a.transpose()


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum(np.dot(v, b) * b for b in basis)
        if (w > 1e-10).any():
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)


# define vars
n_epoch = 100
n_classes = 10
n_features = 16
learning_rate = 0.001
model_path = './models/mse-svd-2.h5'
cluster_centres = np.zeros(shape=(n_classes, n_features))

# Prepare dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)
y_train_vec = to_categorical(y_train)
y_test_vec = to_categorical(y_test)

# create model
input_layer = Input(shape=(28, 28, 1))
conv1 = Conv2D(32, kernel_size=(3, 3),
               activation='relu')(input_layer)
conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
flat1 = Flatten()(pool1)
fc1 = Dense(128, activation='tanh')(flat1)
fc2 = Dense(n_features, activation='tanh')(fc1)
cifar_net = Model(inputs=input_layer, outputs=fc2)
print(cifar_net.summary())

# load model
# cifar_net = load_model(model_path)

adam_op = adam(lr=learning_rate)
cifar_net.compile(optimizer=adam_op,
                  loss=mse,
                  metrics=[mse])

inter_cluster = []
intra_cluster = []

for i in range(n_epoch):

    # get cluster centres in feature space
    for j in range(n_classes):
        j_idx = y_train == j
        x_train_j = x_train[j_idx, :, :, :]
        feat_j = cifar_net.predict_on_batch(x_train_j)
        # print(feat_j.shape)
        cluster_centres[j, :] = np.mean(feat_j, axis=0)

    # ortho-normalize these cluster centres
    # U, temp1, temp2 = svd(cluster_centres.transpose(), full_matrices=False)
    U = gram_schmidt(cluster_centres).transpose()
    # print(np.matmul(U.transpose(), U))
    # print(U.shape)
    cluster_centres = U.transpose()

    # inter cluster distance
    temp = 0
    for j in range(n_classes):
        for k in range(n_classes):
            if k > j:
                temp += norm(cluster_centres[j, :] - cluster_centres[k, :])

    inter_cluster.append(temp)
    # train network to push the features near cluster centres
    feat_train = cluster_centres[y_train, :]
    cifar_net.fit(x=x_train, y=feat_train, batch_size=1024, epochs=1, verbose=0)
    score = cifar_net.evaluate(x_train, feat_train, verbose=0)
    intra_cluster.append(score)
    plt.plot(inter_cluster)
    plt.pause(1)
    plt.clf()
    plt.plot(intra_cluster)
    plt.pause(1)
    plt.close()
    cifar_net.save(model_path, overwrite=True)

embedded_train = cifar_net.predict(x=x_train)
embedded_test = cifar_net.predict(x=x_test)
savetxt(fname='train_feat.txt', X=embedded_train, delimiter=',')
savetxt(fname='test_feat.txt', X=embedded_test, delimiter=',')
savetxt(fname='train_labels', X=y_train, delimiter=',')
savetxt(fname='test_labels', X=y_test, delimiter=',')

