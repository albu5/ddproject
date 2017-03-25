import os
import csv
from random import shuffle

class Dataset:
	def __init__(self, feat, datadir, max_seqlen, split_tol=0.01, split_ratio=2.0/3.0):
		
		class_cat = ['1', '2']		#categorical labels
		class_num = [1,2]			#numerical labels
		
		self.datadir = datadir
		self.feattype = feat
		self.max_seqlen = max_seqlen
		self.seqs_dir = []
		self.seqs_label = []
		self.tbid = 0
		self.vbid = 0

		for category in class_cat:
			class_dir = self.datadir + '/' + category
			temp = ([x[0] for x in os.walk(class_dir)])
			temp.pop(0)
			self.seqs_dir += temp
			if (category == '1'):
				temp2 = [[1., 0.] for x in temp]
			else:
				temp2 = [[0., 1.] for x in temp]

			self.seqs_label += temp2

		split_ratio = 2.0/3.0;

		split_diff = split_tol*2

		while split_diff>split_tol:
			idx = range(len(self.seqs_dir))
			shuffle(idx)

			self.train_idx = idx[:int(split_ratio*len(self.seqs_dir))]
			self.valid_idx = idx[int(split_ratio*len(self.seqs_dir)):]

			print len(self.seqs_label), len(self.seqs_label[0])
			train1 = 0
			train2 = 0
			valid1 = 0
			valid2 = 0
			for a in self.train_idx:
				temp = self.seqs_label[a]
				train1 += temp[0]
				train2 += temp[1]
			for a in self.valid_idx:
				temp = self.seqs_label[a]
				valid1 += temp[0]
				valid2 += temp[1]
			
			# train1 = ([self.seqs_label[a] for a in self.train_idx]).count([1., 0.])
			# train2 = ([self.seqs_label[a] for a in self.train_idx]).count([0., 1.])
			# valid1 = ([self.seqs_label[a] for a in self.valid_idx]).count([1., 0.])
			# valid2 = ([self.seqs_label[a] for a in self.valid_idx]).count([0., 1.])

			split_diff = abs((float(train1)/float(train2)) - (float(valid1)/float(valid2)))
			# print 'split difference is: ', split_diff
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

		print 'total number of training sequences is: ', len(self.train_idx)
		print 'total number of validation sequences is: ', len(self.valid_idx)
		print 'class ration in training set is: ', float(train1)/float(train2)
		print 'class ration in validation set is: ', float(valid1)/float(valid2)

	def getTrainBatch(self, batch_size):
		if (self.tbid+batch_size-1>=len(self.train_idx)):
			tbidx = self.train_idx[:batch_size]
			self.tbid = 0
		else:
			tbidx = self.train_idx[self.tbid:self.tbid+batch_size]
			self.tbid = self.tbid+batch_size

		# print 'tbid', self.tbid, 'of', len(self.train_idx)

		batch_dirs = [self.seqs_dir[a] for a in tbidx]
		batch_labels = [self.seqs_label[a] for a in tbidx]
		batch_data = []
		batch_seqlen = []
		for seq in batch_dirs:
			feat_file = seq+'/'+self.feattype
			with open(feat_file,'rb') as f:
				reader = csv.reader(f)
				temp = list(reader)
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
		if (self.vbid+batch_size-1>=len(self.valid_idx)):
			vbidx = self.valid_idx[:batch_size]
			self.vbid = 0
		else:
			vbidx = self.valid_idx[self.vbid:self.vbid+batch_size]
			self.vbid = self.vbid+batch_size

		# print 'vbid', self.vbid, 'of', len(self.valid_idx)

		batch_dirs = [self.seqs_dir[a] for a in vbidx]
		batch_labels = [self.seqs_label[a] for a in vbidx]
		batch_data = []
		batch_seqlen = []
		for seq in batch_dirs:
			feat_file = seq+'/'+self.feattype
			with open(feat_file,'rb') as f:
				reader = csv.reader(f)
				temp = list(reader)
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
