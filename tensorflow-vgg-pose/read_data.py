from tensorflow.examples.tutorials.mnist import input_data
from numpy import genfromtxt
import tensorflow as tf
from skimage import io
from numpy.random import randint
import numpy as np
from skimage.transform import resize
def read_train(datadir, bs, numlabels):
	datatype = 'train'
	ytemp = genfromtxt(datadir + '/' + datatype + '/labels.txt', delimiter=',')
	yfull = ytemp.astype(int) - 1

	bidx = randint(low = 0, high = yfull.size, size = [bs])
	

	batch = [np.zeros([bs, 224,224,3]), np.zeros([bs, numlabels])]

	counter = 0

	for idx in bidx:

		im1 = io.imread(datadir + '/' + datatype + '/' + ("%d.jpg" % (idx+1)))
		#im1 = resize(im, [100,40])
		imflat = im1
		xidx = imflat.astype(float)

		batch[1][counter][yfull[idx]] = 1
		batch[0][counter] = xidx/255
		counter += 1
	return batch


def read_valid(datadir, numlabels, startidx, endidx):
	datatype = 'valid'
	bs = endidx-startidx

	ytemp = genfromtxt(datadir + '/' + datatype + '/labels.txt', delimiter=',')
	yfull = ytemp.astype(int) - 1


	bidx = np.arange(0,bs)
	

	batch = [np.zeros([bs, 224,224,3]), np.zeros([bs, numlabels])]

	counter = 0

	for idx in range(startidx,endidx):
		im1 = io.imread(datadir + '/' + datatype + '/' + ("%d.jpg" % (idx+1)))
		#im1 = resize(im, [100,40])
		imflat = im1
		xidx = imflat.astype(float)

		batch[1][counter][yfull[idx]] = 1
		batch[0][counter] = xidx/255
		counter += 1
	return batch

