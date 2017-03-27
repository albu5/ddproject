import numpy as np
import os
from random import choice as choose
from random import randint
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

nex = 1
curr_pos = 0


def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_batch(data_dir, batch_size, seq_len):
    global nex
    global curr_pos
    ex_dir = data_dir + '/' + str(nex)
    tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
    tracks_len = tracks.shape[1]
    x = np.empty(shape=(batch_size, seq_len, 4), dtype=np.float32)
    y = np.empty(shape=(batch_size, seq_len, 2), dtype=np.float32)
    good_ex = []
    for i in range(batch_size):
        if tracks_len < seq_len + curr_pos:
            temp_y = 0
            temp_x = 0
            good_ex.append(False)
            nex = randint(1, 639)
            curr_pos = 0
            ex_dir = data_dir + '/' + str(nex)
            print(ex_dir)
            tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
            tracks_len = tracks.shape[1]
        else:
            temp_x = tracks[1:5, curr_pos: curr_pos + seq_len]
            temp_y = np.expand_dims(tracks[5][curr_pos: curr_pos + seq_len], axis=0)
            if np.any(temp_y == 0):
                good_ex.append(False)
            else:
                good_ex.append(True)
            temp_y = np.vstack((temp_y == 1, temp_y == 2))
            curr_pos += seq_len

        if good_ex[-1]:
            x[i, :, :] = temp_x
            y[i, :, :] = np.transpose(temp_y)
    # print(np.array(good_ex))
    x = x[np.array(good_ex), :, :]
    y = y[np.array(good_ex), :, :]
    return x.astype(np.float32), y.astype(np.float32)

def get_batch_resnet(data_dir, batch_size, seq_len):
    global nex
    global curr_pos
    ex_dir = data_dir + '/' + str(nex)
    tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
    tracks_len = tracks.shape[1]
    resnet_features = np.genfromtxt(ex_dir + '/' + 'resnet50.txt', delimiter=',')
    x = np.empty(shape=(batch_size, seq_len, resnet_features.shape[1]), dtype=np.float32)
    y = np.empty(shape=(batch_size, seq_len, 2), dtype=np.float32)
    good_ex = []
    for i in range(batch_size):
        if tracks_len < seq_len + curr_pos:
            temp_y = 0
            temp_x = 0
            good_ex.append(False)
            nex = randint(1, 639)
            curr_pos = 0
            ex_dir = data_dir + '/' + str(nex)
            print(ex_dir)
            tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
            tracks_len = tracks.shape[1]
            resnet_features = np.genfromtxt(ex_dir + '/' + 'resnet50.txt', delimiter=',')
        else:
            temp_x = resnet_features[curr_pos: curr_pos + seq_len, :]
            temp_y = np.expand_dims(tracks[5][curr_pos: curr_pos + seq_len], axis=0)
            if np.any(temp_y == 0):
                good_ex.append(False)
            else:
                good_ex.append(True)
            temp_y = np.vstack((temp_y == 1, temp_y == 2))
            curr_pos += seq_len

        if good_ex[-1]:
            x[i, :, :] = temp_x
            y[i, :, :] = np.transpose(temp_y)
    # print(np.array(good_ex))
    x = x[np.array(good_ex), :, :]
    y = y[np.array(good_ex), :, :]
    return x.astype(np.float32), y.astype(np.float32)


def get_batch_image(data_dir, batch_size, seq_len):
    global nex
    global curr_pos
    ex_dir = data_dir + '/' + str(nex)
    tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
    tracks_len = tracks.shape[1]
    x = np.empty(shape=(batch_size, seq_len, 224*224*3), dtype=np.float32)
    y = np.empty(shape=(batch_size, seq_len, 2), dtype=np.float32)
    good_ex = []
    for i in range(batch_size):
        if tracks_len < seq_len + curr_pos:
            temp_y = 0
            temp_x = 0
            good_ex.append(False)
            nex = randint(1, 639)
            curr_pos = 0
            ex_dir = data_dir + '/' + str(nex)
            print(ex_dir)
            tracks = np.genfromtxt(ex_dir + '/' + 'tracks.txt', delimiter=',')
            tracks_len = tracks.shape[1]
        else:
            temp_x = np.empty(shape=(seq_len, 224, 224, 3))
            ids = tracks[0, curr_pos: curr_pos + seq_len]
            ctr = 0
            for idx in ids:
                im_path = ex_dir + '/' + '%4.4d.jpg' % idx
                im = image.load_img(im_path, target_size=(224, 224))
                im_ = image.img_to_array(im)
                temp_x[ctr, :, :, :] = im_
                ctr += 1
            temp_x = preprocess_input(temp_x)
            temp_x = np.reshape(temp_x, newshape=(seq_len, 224*224*3))
            temp_y = np.expand_dims(tracks[5][curr_pos: curr_pos + seq_len], axis=0)
            if np.any(temp_y == 0):
                good_ex.append(False)
            else:
                good_ex.append(True)
            temp_y = np.vstack((temp_y == 1, temp_y == 2))
            curr_pos += seq_len

        if good_ex[-1]:
            x[i, :, :] = temp_x
            y[i, :, :] = np.transpose(temp_y)
    # print(np.array(good_ex))
    x = x[np.array(good_ex), :, :]
    y = y[np.array(good_ex), :, :]
    return x.astype(np.float32), y.astype(np.float32)

data_dir = './ActivityDataset'
traj_dir = data_dir + '/' + 'TrajectoriesLong'
train_dir = traj_dir + '/' + 'train'

bx, by = get_batch_image(train_dir, 20, 9)
print(bx.shape)
print(by.shape)
print(by)
'''
bx, by = get_batch_resnet(train_dir, 5, 9)
print(bx[1, :, 0:8])
print(np.squeeze(by))


ex_dirs = get_immediate_subdirectories(train_dir)
bx, by = get_batch(ex_dirs, 5, 15)
print(bx[1, :, :])
print(np.squeeze(by))
'''