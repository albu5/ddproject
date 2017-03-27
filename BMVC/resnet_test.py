from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.layers import Input, TimeDistributed, Reshape
from keras.models import Model
from keras.applications.resnet50 import preprocess_input
import numpy as np
import tensorflow as tf
from keras.layers.core import K
K.set_learning_phase(0)

resnet = ResNet50(weights='imagenet', include_top=False)

in_layer = Input(shape=(3, 224*224*3))
im_layer = Reshape(target_shape=(3, 224, 224, 3))(in_layer)
out_layer = TimeDistributed(resnet)(im_layer)
model = Model(inputs=in_layer, outputs=out_layer)
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
x = np.reshape(x, newshape=(1, 224*224*3))
x = np.expand_dims(x, axis=0)
x = np.concatenate([x, x], axis=0)
x = np.concatenate([x, x, x], axis=1)
print(x.shape)

features = model.predict(x)
print(features)
print(features.shape)
