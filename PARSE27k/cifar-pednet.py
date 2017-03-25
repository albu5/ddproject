from tensorflow.examples.tutorials.mnist import input_data
from numpy import genfromtxt
import tensorflow as tf
from skimage import io
from numpy.random import randint
import numpy as np
from skimage.transform import resize
from read_data import read_train
from read_data import read_valid
slim = tf.contrib.slim

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool(x,k,s):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,s,s,1], padding='SAME')

def mean_pool(x,k,s):
	return tf.nn.avg_pool(x, ksize=[1,k,k,1], strides=[1,s,s,1], padding='SAME')


# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


EPS = 1e-8
MAX_STEPS = 20000
BATCH_SIZE = 32
STEP_SIZE = 1e-4
NUM_THREADS = 32
LAMBDA = 0


sess = tf.InteractiveSession()

trainlabels = genfromtxt('PARSE27' + '/' + 'train' + '/labels.txt', delimiter=',')
NUM_LABELS = int(trainlabels.max())
print NUM_LABELS

# building placeholders
x = tf.placeholder(tf.float32, shape = [None, 192*128*3])
y = tf.placeholder(tf.float32, shape = [None, NUM_LABELS])


"""
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# initialize weights and bias
sess.run(tf.initialize_all_variables())

# feedforward op
yff = tf.nn.softmax(tf.matmul(x,W)+b)

# cross entropy loss
loss_cross_entropy = -tf.reduce_sum(y*tf.log(yff))

# select optimizer, minimize cross_entropy wrt variables (W,b)
train_step = tf.train.GradientDescentOptimizer(STEP_SIZE).minimize(loss_cross_entropy)



correct_prediction = tf.equal(tf.argmax(yff,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(MAX_STEPS):
	batch = mnist.train.next_batch(BATCH_SIZE)
	feed_dict = {x: batch[0], y: batch[1]}
	train_step.run(feed_dict)
	print(accuracy.eval(feed_dict={x: mnist.train.images, y: mnist.train.labels}))


print(accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))

"""

#####################CONVNET STARTS HERE#######################
# batch width height channels
x_image = tf.reshape(x, [-1,192,128,3])
keep_prob = tf.placeholder(tf.float32)

# The first two dimensions are the patch size
# the next is the number of input channels
# and the last is the number of output channels
W_conv1 = weight_variable([5,5,3,32])
b_conv1 = weight_variable([32])

# first relu+maxpool layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1,3,2)
h_lrn1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

# second layer
W_conv2 = weight_variable([5,5,32,32])
b_conv2 = weight_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_lrn1, W_conv2) + b_conv2)
h_pool2 = mean_pool(h_conv2,3,2)
h_lrn2 = tf.nn.lrn(h_pool2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')

# third layer
W_conv3 = weight_variable([5,5,32,64])
b_conv3 = weight_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_lrn2, W_conv3) + b_conv3)
h_pool3 = mean_pool(h_conv3,3,2)

# fully connected layer
W_fc1 = weight_variable([24*16*64, 384])
b_fc1 = bias_variable([384])

h_flat = tf.reshape(h_pool3, [-1, 24*16*64])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

# dropout layer
h_drop1 = tf.nn.dropout(h_fc1, keep_prob)

# fc layer 2
W_fc2 = weight_variable([384, 192])
b_fc2 = bias_variable([192])

h_fc2 = tf.nn.relu(tf.matmul(h_drop1, W_fc2) + b_fc2)

# readout layer
W_fc3 = weight_variable([192, NUM_LABELS])
b_fc3 = bias_variable([NUM_LABELS])

y_conv = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)


#####################CONVNET  ENDS  HERE#######################

# define loss
loss_cross_entropy = -tf.reduce_sum(y*tf.log(y_conv+EPS))
# define optimization step
train_step = tf.train.AdamOptimizer(STEP_SIZE).minimize(loss_cross_entropy)

# define accuracy
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables());

tfsaver = tf.train.Saver()
for i in range(MAX_STEPS):
	batch = read_train('PARSE27', BATCH_SIZE, NUM_LABELS)	
	print batch[0].shape
	print batch[1].shape

	# print a log
	if (i%5 == 0)&(i>0):
		train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, 100.0*train_accuracy))

		batch2 = read_valid('PARSE27', NUM_LABELS)
		it = 0
		test_accuracy2 = 0
		while it*BATCH_SIZE<6500:
			test_accuracy2 = (it*test_accuracy2+accuracy.eval(feed_dict={x: batch2[0][it*BATCH_SIZE:(it+1)*BATCH_SIZE-1,:], y: batch2[1][it*BATCH_SIZE:(it+1)*BATCH_SIZE-1,:], keep_prob: 1.0}))/(it+1)
			it = it + 1
		print("step %d, test accuracy %g"%(i, 100.0*test_accuracy2))


		tmpstr = ("./conv3/parse27-conv1.ckpt")
		print tmpstr
		save_path = tfsaver.save(sess, tmpstr)

	# run train step
	train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

batch = read_valid('PARSE27', NUM_LABELS)
test_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
print("step %d, test accuracy %g"%(i, 100.0*test_accuracy))




