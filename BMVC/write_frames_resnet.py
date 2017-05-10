from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from numpy import savetxt
import numpy as np
import tensorflow as tf
import os


with tf.device('/cpu:0'):
    data_dir = './ActivityDataset'
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    for seq in range(1, 45):
        seq_dir = data_dir + '/' + 'seq%2.2d' % seq
        # if not os.path.exists(seq_dir + '/resnet/'):
        #     os.makedirs(seq_dir + '/resnet/')
        if not os.path.exists(seq_dir + '/resnet50.txt'):
            seq_resnet = []
            for frame in range(1, 5000):
                try:
                    im_path = seq_dir + '/' + 'frame%4.4d.jpg' % frame
                    print(im_path)
                    img = image.load_img(im_path, target_size=(224, 224, 3))
                    img = image.img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    img = preprocess_input(img)
                    imf = np.squeeze(resnet.predict_on_batch(img))
                    seq_resnet.append(imf.tolist())
                except:
                    savetxt(seq_dir + '/resnet50.txt', seq_resnet, delimiter=',')
                    break
