"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import utils
from scipy import misc
import os
import numpy as np
from random import shuffle
from readData import dataset

EPS = 1e-8
NUM_EPOCH = 200000
BATCH_SIZE = 8
STEP_SIZE = 1e-4
NUM_THREADS = 32
LAMBDA = 0
NUM_LABELS = 8
VERBOSE = 100.0
datadir = '/home/ashish/Desktop/parse27_codes/PARSE27K224'
classes = [str(i+1) for i in range(8)]
splits = ['train', 'valid', 'test']

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

soft_list = [[1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
			 [0.7, 1.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
			 [0.0, 0.7, 1.0, 0.7, 0.0, 0.0, 0.0, 0.0],
			 [0.0, 0.0, 0.7, 1.0, 0.7, 0.0, 0.0, 0.0],
			 [0.0, 0.0, 0.0, 0.7, 1.0, 0.7, 0.0, 0.0],
			 [0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.7, 0.0],
			 [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.0, 0.7],
			 [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 1.0]]

soft_matrix = tf.constant(soft_list, dtype=tf.float32)

sess = tf.InteractiveSession()

images = tf.placeholder(tf.float32, [BATCH_SIZE, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [BATCH_SIZE, 8])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode)

# print number of variables used: 143667240 variables, i.e. ideal size = 548MB
print vgg.get_var_count()

W_prob = weight_variable([4096, 8])
B_prob = weight_variable([8])

pedprob = tf.nn.softmax(tf.matmul(vgg.relu7,W_prob)+B_prob)

# define loss
loss_cross_entropy = -tf.reduce_sum(tf.matmul(true_out,soft_matrix)*tf.log(pedprob+EPS))
# define optimization step
train_step = tf.train.AdamOptimizer(STEP_SIZE).minimize(loss_cross_entropy)
# define accuracy
correct_prediction = tf.equal(tf.argmax(pedprob,1), tf.argmax(true_out,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

dataset = dataset(datadir, splits, classes)

# sess.run(tf.initialize_all_variables())
tfsaver = tf.train.Saver()
BS = BATCH_SIZE

epoch = 0
it = 0
while epoch<NUM_EPOCH:
	it = it+1
	
	if dataset.tbid == 0:
		epoch += 1

	# print a log
	if (it%(VERBOSE/10) == 0)&(it>0):
		train_accuracy = accuracy.eval(feed_dict={images: data, true_out: labels, train_mode: False})
		print("epoch: %d, step: %d, training accuracy %g"%(epoch, it, 100.0*train_accuracy))

	# print a log
	if (it%(VERBOSE) == 0)&(it>0):
		test_accuracy = 0.0
		data,labels = dataset.getValidBatch(BS)
		counter = 0.0
		while dataset.vbid > 0:
			temp = accuracy.eval(feed_dict={images: data, true_out: labels, train_mode: False})
			test_accuracy = (counter*test_accuracy + temp)/(counter+1)
			counter += 1
		print("epoch: %d, step: %d, validation accuracy %g"%(epoch, it, 100.0*test_accuracy))
		'''
		batch2 = read_valid('../PARSE224', NUM_LABELS)
		
		it = 0
		test_accuracy2 = 0
		while it*BS<6500:
			test_accuracy2 = (it*test_accuracy2+accuracy.eval(feed_dict={images: batch2[0][it*BS:((it+1)*BS),:], true_out: batch2[1][it*BS:((it+1)*BS),:], train_mode:False}))/(it+1)
			it = it + 1
		print("step %d, test accuracy %g"%(i, 100.0*test_accuracy2))
		'''
		tmpstr = ("./models/PARSE27-VGG19-SOFT-%d.ckpt"%(epoch))
		save_path = tfsaver.save(sess, tmpstr)

	# run train step
	data,labels = dataset.getTrainBatch(BS)
	train_step.run(feed_dict={images: data, true_out: labels, train_mode:True})

'''
# test classification
prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
utils.print_prob(prob[0], './synset.txt')

# simple 1-step training
cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

# test classification again, should have a higher probability about tiger
prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
utils.print_prob(prob[0], './synset.txt')

# test save
vgg.save_npy(sess, './test-save.npy')
'''