from tensorflow.examples.tutorials.mnist import input_data
from numpy import genfromtxt
import tensorflow as tf
from skimage import io
from numpy.random import randint
import numpy as np
from skimage.transform import resize
from read_data import read_train
from read_data import read_valid


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


EPS = 1e-8
MAX_STEPS = 20000
BATCH_SIZE = 64
STEP_SIZE = 1e-4
NUM_THREADS = 32


sess = tf.InteractiveSession()

trainlabels = genfromtxt('MULTIVIEW96x40' + '/' + 'train' + '/labels.txt', delimiter=',')
NUM_LABELS = int(trainlabels.max())
print NUM_LABELS

# building placeholders
x = tf.placeholder(tf.float32, shape = [None, 96*40])
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
x_image = tf.reshape(x, [-1,96,40,1])


# The first two dimensions are the patch size
# the next is the number of input channels
# and the last is the number of output channels
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = weight_variable([32])

# first relu+maxpool layer
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second conv+maxpool layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = weight_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# second conv+maxpool layer
W_conv3 = weight_variable([12,5,64,128])
b_conv3 = weight_variable([128])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


# fully connected layer
W_fc1 = weight_variable([12*5*128, 512])
b_fc1 = bias_variable([512])

h_pool3_flattened = tf.reshape(h_pool3, [-1, 12*5*128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flattened, W_fc1) + b_fc1)

# dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# readout layer
W_fc2 = weight_variable([512, NUM_LABELS])
b_fc2 = bias_variable([NUM_LABELS])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
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
	batch = read_train('MULTIVIEW96x40', BATCH_SIZE, NUM_LABELS)	
	
	# print a log
	if (i%20 == 0)&(i>0):
		train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, 100.0*train_accuracy))

		batch2 = read_valid('MULTIVIEW96x40', NUM_LABELS)
		test_accuracy2 = accuracy.eval(feed_dict={x: batch2[0], y: batch2[1], keep_prob: 1.0})
		print("step %d, test accuracy %g"%(i, 100.0*test_accuracy2))


		tmpstr = ("./conv3/dev-%d.ckpt"%(i))
		print tmpstr
		save_path = tfsaver.save(sess, tmpstr)

	# run train step
	train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

batch = read_valid('MULTIVIEW96x40', NUM_LABELS)
test_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
print("step %d, test accuracy %g"%(i, 100.0*test_accuracy))



