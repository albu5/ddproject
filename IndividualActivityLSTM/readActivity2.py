import os
import csv
from random import shuffle
import random

class Dataset:
	def __init__(self, feat, datadir, max_seqlen):
		
		class_cat = ['1', '2']		#categorical labels
		class_num = [1,2]			#numerical labels
		
		self.datadir = datadir
		self.feattype = feat
		self.max_seqlen = max_seqlen

		self.seqs_dir_train = []
		self.seqs_label_train = []
		self.seqs_dir_valid = []
		self.seqs_label_valid = []
		
		self.tbid = 0
		self.vbid = 0

		for category in class_cat:
			class_dir = self.datadir + '/train/' + category
			temp = ([x[0] for x in os.walk(class_dir)])
			temp.pop(0)
			self.seqs_dir_train += temp
			if (category == '1'):
				temp2 = [[1., 0.] for x in temp]
			else:
				temp2 = [[0., 1.] for x in temp]

			self.seqs_label_train += temp2

		# shuffle trainig data
		idx = range(len(self.seqs_dir_train))
		shuffle(idx)
		self.seqs_dir_train = [self.seqs_dir_train[i] for i in idx]
		self.seqs_label_train = [self.seqs_label_train[i] for i in idx]

		for category in class_cat:
			class_dir = self.datadir + '/valid/' + category
			temp = ([x[0] for x in os.walk(class_dir)])
			temp.pop(0)
			self.seqs_dir_valid += temp
			if (category == '1'):
				temp2 = [[1., 0.] for x in temp]
			else:
				temp2 = [[0., 1.] for x in temp]

			self.seqs_label_valid += temp2

		# i = 0
		# for seq in seqs:
		# 	feat_file = seq+'/'+feat
		# 	with open(feat_file,'rb') as f:
		# 		reader = csv.reader(f)
		# 		curr_seq = list(reader)
		# 		for vec in curr_seq:
		# 			print vec
		# 		print 'feature vector length is: ', len(curr_seq)
		# 	break
		# 	i = i+1
		train1 = self.seqs_label_train.count([1., 0.])
		train2 = self.seqs_label_train.count([0., 1.])
		valid1 = self.seqs_label_valid.count([1., 0.])
		valid2 = self.seqs_label_valid.count([0., 1.])

		
		print 'total number of training sequences is: ', len(self.seqs_dir_train)
		print 'total number of validation sequences is: ', len(self.seqs_dir_valid)
		print 'class ration in training set is: ', float(train1)/float(train2)
		print 'class ration in validation set is: ', float(valid1)/float(valid2)

	def getTrainBatch(self, batch_size):
		if (self.tbid+batch_size-1>=len(self.seqs_dir_train)):
			batch_dirs = self.seqs_dir_train[:batch_size]
			batch_labels = self.seqs_label_train[:batch_size]
			self.tbid = 0
		else:
			batch_dirs = self.seqs_dir_train[self.tbid:self.tbid+batch_size]
			batch_labels = self.seqs_label_train[self.tbid:self.tbid+batch_size]			
			self.tbid = self.tbid+batch_size

		# print 'tbid', self.tbid, 'of', len(self.train_idx)

		batch_data = []
		batch_seqlen = []
		for seq in batch_dirs:
			feat_file = seq+'/'+self.feattype
			with open(feat_file,'rb') as f:
				reader = csv.reader(f)
				temp = list(reader)
				temp = [vec[-4:] for vec in temp]
				batch_seqlen.append(len(temp))
				padsize = self.max_seqlen-len(temp)
				padarr = [[0 for x in temp[0]] for y in range(padsize)]
				temp += padarr
				batch_data.append(temp)
				
				# for vec in curr_seq:
				# 	print vec
				#print 'current feature vector length is: ', len(temp)
			# break
		return batch_data, batch_labels, batch_seqlen

	def getValidBatch(self, batch_size):
		if (self.vbid+batch_size-1>=len(self.seqs_dir_valid)):
			batch_dirs = self.seqs_dir_valid[:batch_size]
			batch_labels = self.seqs_label_valid[:batch_size]
			self.vbid = 0
		else:
			batch_dirs = self.seqs_dir_valid[self.vbid:self.vbid+batch_size]
			batch_labels = self.seqs_label_valid[self.vbid:self.vbid+batch_size]			
			self.vbid = self.vbid+batch_size

		# print 'vbid', self.vbid, 'of', len(self.seqs_dir_valid)

		batch_data = []
		batch_seqlen = []
		for seq in batch_dirs:
			feat_file = seq+'/'+self.feattype
			with open(feat_file,'rb') as f:
				reader = csv.reader(f)
				temp = list(reader)
				temp = [vec[-4:] for vec in temp]
				batch_seqlen.append(len(temp))
				padsize = self.max_seqlen-len(temp)
				padarr = [[0 for x in temp[0]] for y in range(padsize)]
				temp += padarr
				batch_data.append(temp)
				
				# for vec in curr_seq:
				# 	print vec
				#print 'current feature vector length is: ', len(temp)
			# break
		return batch_data, batch_labels, batch_seqlen



# feat = 'hog.csv'
# datadir = '/home/ashish/Desktop/ped-det/data/ActivityDataset/individuals_short'
# max_seqlen = 91
# split_tol = 0.03
# Dataset = Dataset(feat, datadir, max_seqlen)

# for i in range(50):
# 	[a,b,c] = Dataset.getTrainBatch(32)
# 	print len(a), len(a[0]), len(a[0][0])
