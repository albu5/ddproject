from keras.models import Model
from keras.layers import Input, Dense

from keras.layers.merge import add
from keras.optimizers import adam
import keras.backend as kb
import numpy as np
from scipy import io
from utils import get_group_instance
from matplotlib import pyplot as plt
from keras import losses
from utils import get_feature_vector


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def kernel_loss(y_true, y_pred):
    inclusion_dist = kb.max(y_pred - 1 + y_true)
    exclusion_dist = kb.max(y_pred - y_true)
    exclusion_dist2 = kb.mean(y_pred * (1 - y_true) * kb.cast(y_pred > 0, dtype=kb.floatx()))

    # ex_cost = kb.log(exclusion_dist + kb.epsilon()) * (1 - kb.prod(y_true))
    # in_cost = -kb.log(inclusion_dist + kb.epsilon()) * (1 - kb.prod(1 - y_true))
    ex_cost = (exclusion_dist + kb.epsilon()) * (1 - kb.prod(y_true))
    in_cost = -(inclusion_dist + kb.epsilon()) * (1 - kb.prod(1 - y_true))
    # return inclusion_dist * kb.sum(y_true)
    # return - exclusion_dist * (1 - kb.prod(y_true))
    return in_cost + 1.2 * ex_cost


def simple_loss(y_true, y_pred):
    res_diff = (y_true - y_pred) * kb.cast(y_pred >= 0, dtype=kb.floatx())
    return kb.sum(kb.square(res_diff))


def process_batch(datum, max_people):
    features = datum['features']
    labels = datum['labels']
    batch_x = []
    batch_y = []
    batch_pad = []
    for member in range(labels.shape[0]):
        pair_feat = []
        pair_membership = []
        pair_pad = []
        for agent in range(labels.shape[0]):
            pair_feat.append(np.hstack((features[member], features[agent])).tolist())
            pair_membership.append(labels[member] == labels[agent])
            if member == agent:
                pair_pad.append(-2)
            else:
                pair_pad.append(0)
        n_remaining = max_people - len(pair_membership)
        for dummy in range(n_remaining):
            pair_feat.append(pair_feat[0])
            pair_membership.append(pair_membership[0])
            pair_pad.append(-2)

        batch_x.append(np.array(pair_feat).astype(np.float32))
        batch_y.append(np.expand_dims(np.array(pair_membership).astype(np.float32), axis=1))
        batch_pad.append(np.expand_dims(np.array(pair_pad).astype(np.float32), axis=1))

    return batch_x, batch_y, batch_pad


def process_batch_custom(datum, max_people):
    features = datum['features']
    labels = datum['labels']
    batch_x = []
    batch_y = []
    batch_pad = []
    for member in range(labels.shape[0]):
        pair_feat = []
        pair_membership = []
        pair_pad = []
        for agent in range(labels.shape[0]):
            pair_feat.append(get_feature_vector(features[member], features[agent]))
            pair_membership.append(labels[member] == labels[agent])
            if member == agent:
                pair_pad.append(-2)
            else:
                pair_pad.append(0)
        n_remaining = max_people - len(pair_membership)
        for dummy in range(n_remaining):
            pair_feat.append(pair_feat[0])
            pair_membership.append(pair_membership[0])
            pair_pad.append(-2)

        batch_x.append(np.array(pair_feat).astype(np.float32))
        batch_y.append(np.expand_dims(np.array(pair_membership).astype(np.float32), axis=1))
        batch_pad.append(np.expand_dims(np.array(pair_pad).astype(np.float32), axis=1))

    return batch_x, batch_y, batch_pad

'''
======================CONSTANTS==================================================================================
'''
losses.simple_loss = simple_loss
losses.kernel_loss = kernel_loss

max_iter = 5000
learning_rate = 0.0001
save_step = 200
disp_step = 100
debug_step = 100
model_path = './models/cad-kernel-affinity-bottom-max-long-custom-20.h5'
n_hidden1 = 2**5
n_hidden2 = 2**5
n_max = 20
f_len = 5


in1 = Input(shape=(f_len*2, ))
fc1 = Dense(n_hidden1, activation='tanh')(in1)
fc2 = Dense(n_hidden2, activation='tanh')(fc1)
sim1 = Dense(1, activation='sigmoid')(fc2)
sim_net = Model(inputs=in1, outputs=sim1)

pad_idx = Input(batch_shape=(n_max, 1))
inp_vec = Input(batch_shape=(n_max, f_len*2))
sim_vec = sim_net(inp_vec)
sim_vec_p = add([sim_vec, pad_idx])   # ignores padding during min
kernel_net = Model(inputs=[inp_vec, pad_idx], outputs=sim_vec_p)

print(kernel_net.summary())
optimizer = adam(lr=learning_rate)
kernel_net.compile(optimizer=optimizer, loss=kernel_loss)

group_data = io.loadmat('../group_data_bottom_long_feat.mat')['trainX'][:, 0]
n_examples = group_data.shape[0]

# data = get_group_instance(group_data)
# print(data['labels'])
# bx, by, bp = process_batch(data, n_max)
# print(len(bx), len(by), len(bp))
# print(bx[0].shape)
# print(bx[0])
# print(by[0])
# print(bp[0])

score_arr = []

for i in range(max_iter):
    data = get_group_instance(group_data)
    if data['labels'].shape[0] > 20:
        continue
    bx, by, bp = process_batch_custom(data, n_max)
    score = 0
    for j in range(len(bx)):
        kernel_net.train_on_batch(x=[bx[j], bp[j]], y=by[j])
        score = (score * j + kernel_net.evaluate(x=[bx[j], bp[j]], y=by[j], verbose=0)) / (j + 1)
    score_arr.append(score)

    if i % debug_step == 0:
        print(data['labels'])
        n_ped = data['labels'].shape[0]
        for j in range(len(bx)):
            temp = np.squeeze(kernel_net.predict_on_batch(x=[bx[j], bp[j]]))
            print()
            print(np.round(temp[0:n_ped], 2))
            print(by[j][0:n_ped, 0])
            print()

    if i % disp_step == 0:
        plt.plot(moving_average(score_arr, 1))
        plt.pause(0.01)

    if i % save_step == 0:
        kernel_net.save(model_path, overwrite=True)


