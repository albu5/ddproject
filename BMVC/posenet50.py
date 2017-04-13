from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Input, Reshape
from keras.models import Model
from keras.optimizers import adam
import os
from numpy import genfromtxt
from matplotlib import pyplot as plt
from utils import get_image_names_and_labels, get_parse_batch
from random import randint
from keras import backend as Keras
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
model_path = './models-posenet50/total_loss_4'


'''
========================================================================================================================
CUSTOM LOSSES HERE
========================================================================================================================
'''


def good_loss(y_true, y_pred):
    return Keras.mean(-y_true * Keras.log(y_pred + Keras.epsilon()), axis=-1)


def bad_loss(y_true, y_pred):
    cost_matrix_np = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                               [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                               ], dtype=np.float32)
    cost_matrix = Keras.constant(value=cost_matrix_np, dtype=Keras.floatx())
    bad_pred = Keras.dot(y_true, cost_matrix)
    return Keras.mean(-bad_pred * Keras.log(1 - y_pred + Keras.epsilon()), axis=-1)


def total_loss(y_true, y_pred):
    return bad_loss(y_true, y_pred) + good_loss(y_true, y_pred)


'''
========================================================================================================================
MODEL DEFINITION
========================================================================================================================
'''
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
input_layer = Input(shape=(224, 224, 3))
resnet_features = resnet(input_layer)
# resnet_flat = Keras.squeeze(Keras.squeeze(resnet_features, axis=1), axis=1)
resnet_features = Reshape(target_shape=(2048, ))(resnet_features)
resnet_dense = Dense(1024, activation='relu')(resnet_features)
resnet_prob = Dense(8, activation='softmax')(resnet_dense)
pose_resnet = Model(input=input_layer, output=resnet_prob)

optimizer = adam(lr=0.0001)
pose_resnet.compile(optimizer=optimizer, loss=total_loss, metrics=['accuracy'])
print(pose_resnet.summary())


'''
========================================================================================================================
TRAINING
========================================================================================================================
'''
data_path = './../tf-vgg/PARSE224/train'
img_names, labels_vec = get_image_names_and_labels(data_path)

x, y = get_parse_batch(batch_size, img_names, labels_vec, 224, 224)
print(x.shape, y.shape)
# plt.imshow((x[0, :, :, :] - x.min())/(x.max() - x.min()))
# plt.pause(2)


acc_arr = []
for i in range(num_iter):
    X, Y = get_parse_batch(batch_size, img_names, labels_vec, 224, 224)
    print('trained on ', i*batch_size, ' images')
    pose_resnet.train_on_batch(X, Y)

    if i % eval_step == 0:
        score, acc = pose_resnet.evaluate(X, Y, batch_size=batch_size, verbose=1)
        acc_arr.append(acc)
        print('Step:', i, 'Score: ', score, 'Accuracy:', acc)

    if (i % decay_step == 0) and i is not 0:
        Keras.set_value(optimizer.lr, 0.5 * Keras.get_value(optimizer.lr))

    if (i % disp_step == 0) and i is not 0:
        plt.plot(acc_arr)
        plt.pause(0.0005)

    if (i % save_step == 0) and i is not 0:
        pose_resnet.save(model_path, overwrite=True)
