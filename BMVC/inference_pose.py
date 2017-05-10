from keras.models import load_model
from utils import get_image_names_and_labels, get_parse_batch
from keras import backend as Keras
from matplotlib import pyplot as plt
from keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import keras.losses
from numpy import genfromtxt, savetxt
from matplotlib import pyplot as plt


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

keras.losses.total_loss = total_loss
'''
========================================================================================================================
RE - TRAINING
========================================================================================================================
'''
data_dir = './ActivityDataset'
anno_dir = data_dir + '/' + 'csvanno'
pose_resnet = load_model(model_path)

permute = [3, 2, 1, 8, 7, 6, 5, 4]
permute = [val-1 for val in permute]


for seq in range(1, 45):
    if seq is not 39:
        meta_vec = []
        pose_vec = []
        seq_dir = data_dir + '/' + 'seq%2.2d' % seq
        anno_path = anno_dir + '/' + 'data_%2.2d.txt' % seq
        anno = genfromtxt(anno_path, delimiter=',')
        for t in range(1, int(np.max(anno[:, 0])+1)):
            print(seq_dir + '/' + 'frame%4.4d.jpg' % t)
            im = Image.open(seq_dir + '/' + 'frame%4.4d.jpg' % t)
            anno_t = anno[anno[:, 0] == t, :]
            if len(anno_t.shape) == 2:
                for ped in anno_t:
                    time = ped[0]
                    idx = ped[1]
                    bb = ped[2:6]
                    bb[2] = bb[0] + bb[2]
                    bb[3] = bb[1] + bb[3]
                    region = im.crop((bb[0], bb[1], bb[2], bb[3]))
                    region = region.resize((224, 224), Image.ANTIALIAS)
                    im_arr = np.fromstring(region.tobytes(), dtype=np.uint8)
                    im_arr = im_arr.reshape((region.size[1], region.size[0], 3))
                    img = np.expand_dims(im_arr, axis=0).astype(np.float32)
                    img = preprocess_input(img)
                    imf = np.squeeze(pose_resnet.predict_on_batch(img))
                    meta_vec.append([seq, time, idx])
                    pose_vec.append(imf[permute].tolist())
                    # plt.imshow(im_arr)
                    # plt.title([('%1.2f' % val) for val in imf[permute]])
                    # plt.pause(1)
            else:
                print('single worked ...............................')
                ped = anno_t
                time = ped[0]
                idx = ped[1]
                bb = ped[2:6]
                bb[2] = bb[0] + bb[2]
                bb[3] = bb[1] + bb[3]
                region = im.crop((bb[0], bb[1], bb[2], bb[3]))
                region = region.resize((224, 224), Image.ANTIALIAS)
                im_arr = np.fromstring(region.tobytes(), dtype=np.uint8)
                im_arr = im_arr.reshape((region.size[1], region.size[0], 3))
                img = np.expand_dims(im_arr, axis=0).astype(np.float32)
                img = preprocess_input(img)
                imf = np.squeeze(pose_resnet.predict_on_batch(img))
                meta_vec.append([seq, time, idx])
                pose_vec.append(imf[permute].tolist())
                plt.imshow(im_arr)
                plt.title([('%1.2f' % val) for val in imf[permute]])
                plt.pause(1)
        print(np.array(pose_vec).shape)
        print(np.array(meta_vec).shape)
        savetxt('./common/pose/pose%2.2d.txt' % seq, np.array(pose_vec), delimiter=',')
        savetxt('./common/pose/meta%2.2d.txt' % seq, np.array(meta_vec), delimiter=',')

savetxt('./common/pose_vec.txt', np.array(pose_vec), delimiter=',')
savetxt('./common/pose_meta.txt', np.array(meta_vec), delimiter=',')





