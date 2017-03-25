
from scipy import misc
import os
import numpy as np
from random import shuffle


class dataset:
	"""docstring for ClassName"""
	def __init__(self, datadir, splits, classes):
		self.datadir = datadir
		self.classes = classes
		self.splits = splits
		self.tbid = 0
		self.vbid = 0
		self.testbid = 0

		if (len(splits)>0):
			self.traindata = []
			self.trainlabels = []
			for label in self.classes:
				classdir = datadir + '/' + 'train' + '/' + label
				print(classdir)
				imgs = os.listdir(classdir)
				for img in imgs:
					impath = classdir + '/' + img
					print(impath)
					im = misc.imread(impath)
					im_ = im.astype(np.float32)/255
					im__ = im_.tolist()
					self.traindata.append(im__)
					logit = [(i==int(label)) for i in range(len(self.classes))]
					self.trainlabels.append(logit)
			idx = [i for i in range(len(self.trainlabels))]
			shuffle(idx)
			self.traindata = [self.traindata[i] for i in idx]
			self.trainlabels = [self.trainlabels[i] for i in idx]

		if (len(splits)>1):
			self.validdata = []
			self.validlabels = []
			for label in self.classes:
				classdir = datadir + '/' + 'valid' + '/' + label
				print(classdir)
				imgs = os.listdir(classdir)
				for img in imgs:
					impath = classdir + '/' + img
					print(impath)
					im = misc.imread(impath)
					im_ = im.astype(np.float32)/255
					im__ = im_.tolist()
					self.validdata.append(im__)
					logit = [(i==int(label)) for i in range(len(self.classes))]
					self.validlabels.append(logit)
			idx = [i for i in range(len(self.validlabels))]
			shuffle(idx)
			self.validdata = [self.validdata[i] for i in idx]
			self.validlabels = [self.validlabels[i] for i in idx]

		if (len(splits)>2):
			self.testdata = []
			self.testlabels = []
			for label in self.classes:
				classdir = datadir + '/' + 'test' + '/' + label
				print(classdir)
				imgs = os.listdir(classdir)
				for img in imgs:
					impath = classdir + '/' + img
					print(impath)
					im = misc.imread(impath)
					im_ = im.astype(np.float32)/255
					im__ = im_.tolist()
					self.testdata.append(im__)
					logit = [(i==int(label)) for i in range(len(self.classes))]
					self.testlabels.append(logit)
			idx = [i for i in range(len(self.testlabels))]
			shuffle(idx)
			self.testdata = [self.testdata[i] for i in idx]
			self.testlabels = [self.testlabels[i] for i in idx]

	def getTrainBatch(self,batch_size):
		if (self.tbid+batch_size-1>=len(self.trainlabels)):
			batch_data = self.traindata[:batch_size]
			batch_labels = self.trainlabels[:batch_size]
			self.tbid = 0
		else:
			batch_data = self.traindata[self.tbid:batch_size+self.tbid]
			batch_labels = self.trainlabels[self.tbid:batch_size+self.tbid]
			self.tbid += batch_size
		return batch_data, batch_labels


	def getValidBatch(self,batch_size):
		if (self.vbid+batch_size-1>=len(self.trainlabels)):
			batch_data = self.validdata[:batch_size]
			batch_labels = self.validlabels[:batch_size]
			self.vbid = 0
		else:
			batch_data = self.validdata[self.vbid:batch_size+self.vbid]
			batch_labels = self.validlabels[self.vbid:batch_size+self.vbid]
			self.vbid += batch_size
		return batch_data, batch_labels

	

	def getTestBatch(self,batch_size):
		if (self.testbid+batch_size-1>=len(self.testlabels)):
			batch_data = self.testdata[:batch_size]
			batch_labels = self.testlabels[:batch_size]
			self.testbid = 0
		else:
			batch_data = self.testdata[self.testbid:batch_size+self.testbid]
			batch_labels = self.testlabels[self.testbid:batch_size+self.testbid]
			self.testbid += batch_size
		return batch_data, batch_labels

	

		