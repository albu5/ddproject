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
	
	tempim = resize(io.imread(datadir + '/' + datatype + '/' + ("%d.png" % (1))), [100, 40])

	batch = [np.zeros([bs, tempim.size]), np.zeros([bs, numlabels])]

	counter = 0

	for idx in bidx:

		im = io.imread(datadir + '/' + datatype + '/' + ("%d.png" % (idx+1)))
		im1 = resize(im, [100,40])
		imflat = im1.reshape([im1.size])
		xidx = imflat.astype(float)

		batch[1][counter][yfull[idx]] = 1
		batch[0][counter] = xidx
		counter += 1
	return batch


def read_valid(datadir, numlabels):
	datatype = 'valid'
	bs = 290

	ytemp = genfromtxt(datadir + '/' + datatype + '/labels.txt', delimiter=',')
	yfull = ytemp.astype(int) - 1


	bidx = np.arange(0,bs)
	
	tempim = resize(io.imread(datadir + '/' + datatype + '/' + ("%d.png" % (1))), [100, 40])

	batch = [np.zeros([bs, tempim.size]), np.zeros([bs, numlabels])]

	counter = 0

	for idx in bidx:

		im = io.imread(datadir + '/' + datatype + '/' + ("%d.png" % (idx+1)))
		im1 = resize(im, [100,40])
		imflat = im1.reshape([im1.size])
		xidx = imflat.astype(float)

		batch[1][counter][yfull[idx]] = 1
		batch[0][counter] = xidx
		counter += 1
	return batch

