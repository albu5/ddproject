from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import adam
from keras.losses import mean_squared_error as mse
from numpy.linalg import norm
from matplotlib import pyplot as plt
from utils import get_group_instance, gram_schmidt
from scipy import io
import numpy as np


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

'''
======================CONSTANTS==================================================================================
'''
n_epoch = 100000
n_features = 2**4
n_hidden = 2**5
learning_rate = 0.001
save_step = 200
disp_step = 100
debug_step = 500
model_path = './models/cad-sdc-2.h5'


'''
======================MODEL DEFINITION==================================================================================
'''
input_layer = Input(shape=(26,))
fc1 = Dense(n_hidden, activation='tanh')(input_layer)
fc2 = Dense(n_features, activation='tanh')(fc1)
group_net = Model(inputs=input_layer, outputs=fc2)
print(group_net.summary())
optimizer = adam(lr=learning_rate)
group_net.compile(optimizer=optimizer, loss=mse, metrics=[mse])


'''
======================TRAINING==================================================================================
'''
group_data = io.loadmat('../group_data.mat')['trainX'][:, 0]
n_examples = group_data.shape[0]

inter_cluster = []
intra_cluster = []
drop_count = [0]

for i in range(n_epoch):
    data = get_group_instance(group_data)
    features = data['features']
    labels = data['labels'] - 1
    n_classes = np.max(labels) + 1

    for j in range(n_classes):
        if np.sum(labels == j) == 0:
            labels[labels > j] = labels[labels > j] - 1
    n_classes = np.max(labels) + 1
    # print(labels, n_classes)

    if labels.shape[0] > 1:
        # calculate cluster centres
        cluster_centres = np.zeros(shape=(n_classes, n_features))
        for j in range(n_classes):
            j_idx = labels == j
            features_j = features[j_idx, :]
            if len(features_j.shape) < 2:
                features_j = np.expand_dims(features_j, axis=0)

            phi_j = group_net.predict(features_j)
            cluster_centres[j, :] = np.mean(phi_j, axis=0)

        # orthonormalize cluster centres
        cluster_centres, dropped, fatal = gram_schmidt(cluster_centres)
        if dropped:
            drop_count.append(drop_count[-1] + 1)
        else:
            drop_count.append(drop_count[-1])

        if fatal:
            continue

        # inter cluster distance
        temp = 0
        count = 0
        for j in range(n_classes):
            for k in range(n_classes):
                if k > j:
                    temp += norm(cluster_centres[j, :] - cluster_centres[k, :])
                    count += 1
        inter_cluster.append(temp/(count + 1e-8))

        if i % debug_step == 0:
            phi = group_net.predict(x=features)
            print(labels)
            N = labels.shape[0]
            pair_dist = np.zeros(shape=(N, N))
            for k in range(N):
                for j in range(N):
                    pair_dist[k, j] = norm(phi[k, :] - phi[j, :])
            print(pair_dist)
            feat_target = cluster_centres[labels, :]
            score = group_net.evaluate(x=features, y=feat_target, verbose=0)
            print(score)

        # train network to push features to cluster centres
        feat_target = cluster_centres[labels, :]
        score = group_net.evaluate(x=features, y=feat_target, verbose=0)
        group_net.train_on_batch(x=features, y=feat_target)
        intra_cluster.append(n_features*score[0])

    if i % disp_step == 0 and i is not 0:
        # plot the results
        plt.clf()
        plt.plot(np.array(drop_count)/np.array(range(1, i+3)))
        plt.pause(1)

        plt.clf()
        plt.plot(moving_average(intra_cluster, 3))
        axes = plt.gca()
        axes.set_ylim([0, 1])
        plt.pause(0.1)

    if i % save_step == 0:
        group_net.save(model_path, overwrite=True)

