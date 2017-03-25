from numpy import genfromtxt
from numpy.random import randint
import numpy as np

def read_train(datadir, bs, numlabels):
	datatype = 'train'

	ytemp = genfromtxt(datadir + '/' + datatype + '/labels.txt', delimiter=',')
	yfull = ytemp.astype(int) - 1

	xtemp = genfromtxt(datadir + '/' + datatype + '/channels.txt', delimiter=',')

	bidx = randint(low = 0, high = yfull.size, size = [bs])

	batch = [np.zeros([bs, 96*40*8]), np.zeros([bs, numlabels])]

	counter = 0

	for idx in bidx:
		imflat = xtemp[idx]
		xidx = imflat.astype(float)

		batch[1][counter][yfull[idx]] = 1
		batch[0][counter] = xidx/255
		counter += 1
	return batch


def read_valid(datadir, numlabels):
	datatype = 'valid'
	bs = 290

	ytemp = genfromtxt(datadir + '/' + datatype + '/labels.txt', delimiter=',')
	yfull = ytemp.astype(int) - 1

	xtemp = genfromtxt(datadir + '/' + datatype + '/channels.txt', delimiter=',')

	bidx = np.arange(0,bs)
	

	batch = [np.zeros([bs, 96*40*8]), np.zeros([bs, numlabels])]

	counter = 0

	for idx in bidx:
		imflat = xtemp[idx]
		xidx = imflat.astype(float)

		batch[1][counter][yfull[idx]] = 1
		batch[0][counter] = xidx/255
		counter += 1
	return batch

