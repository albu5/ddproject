import numpy as np
import os
from random import choice as choose
from numpy import genfromtxt
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
            # print(ex_dir)
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
            x[i, :, :] = np.transpose(temp_x)
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


def get_image_names_and_labels(data_dir):
    labels_ind = genfromtxt(os.path.join(data_dir, 'labels.txt'))-1
    labels = np.zeros((labels_ind.shape[0], 8))
    labels[np.arange(labels_ind.shape[0]).astype(np.int), labels_ind.astype(np.int)] = 1
    img_list = []
    for i in range(labels.shape[0]):
        img_list.append(os.path.join(data_dir, '%d.jpg' % (i + 1)))
    return img_list, labels


def get_parse_batch(batch_size, img_list, labels, height, width, n_classes=8):
    n_files = len(img_list)
    batch_images = np.zeros((batch_size, height, width, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, n_classes), dtype=np.float32)

    # good_ids = []
    for i in range(batch_size):
        idx = randint(0, n_files - 1)
        img = image.load_img(img_list[idx], target_size=(224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        labels_i = labels[idx, :]
        batch_images[i, :, :, :] = img
        batch_labels[i, :] = labels_i
    return batch_images, batch_labels
