from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import numpy as np
import csv
from matplotlib import pyplot as plt

model = ResNet50(weights='imagenet', include_top=False)

data_dir = './ActivityDataset/TrajectoriesLong'
splits = ['train', 'test']


for split in splits:
    split_dir = data_dir + '/' + split
    nex = 1
    while True:
        try:
            ex_dir = split_dir + '/' + str(nex)
            track_file = ex_dir + '/' + 'tracks.txt'
            tracks = np.genfromtxt(track_file, delimiter=',')
            print(tracks.shape)
            ids = tracks[0][:]
            f_arr = []
            for idx in ids:
                im_path = ex_dir + '/' + '%4.4d.jpg' % idx
                print(im_path)
                im = image.load_img(im_path, target_size=(224, 224))
                x = image.img_to_array(im)
                # plt.imshow(im)
                # plt.pause(2)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                features = np.squeeze(model.predict(x)).tolist()
                f_arr.append(features)
            f_path = ex_dir + '/' + 'resnet50.txt'
            with open(f_path, "w") as f:
                writer = csv.writer(f)
                writer.writerows(f_arr)
            nex += 1
        except:
            break

