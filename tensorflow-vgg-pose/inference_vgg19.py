"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf

import vgg19_trainable as vgg19
import utils
from read_data import read_train
from read_data import read_valid
import glob, os
import csv
from scipy import misc
import matplotlib.pyplot as plt
from numpy import float32
from PIL import Image
import numpy

EPS = 1e-8
MAX_STEPS = 200000
BATCH_SIZE = 32
STEP_SIZE = 1e-4
NUM_THREADS = 32
LAMBDA = 0
NUM_LABELS = 8

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)

config = tf.ConfigProto()
config.gpu_options.allocator_type='BFC'
sess = tf.InteractiveSession()

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
true_out = tf.placeholder(tf.float32, [None, 8])
train_mode = tf.placeholder(tf.bool)

vgg = vgg19.Vgg19('./vgg19.npy')
vgg.build(images, train_mode)

# print number of variables used: 143667240 variables, i.e. ideal size = 548MB
print "model loaded, var count is ", vgg.get_var_count()


W_prob = weight_variable([4096, 8])
B_prob = weight_variable([8])

pedprob = tf.nn.softmax(tf.matmul(vgg.relu7,W_prob)+B_prob)

# define loss
loss_cross_entropy = -tf.reduce_sum(true_out*tf.log(pedprob+EPS))
# define optimization step
train_step = tf.train.AdamOptimizer(STEP_SIZE).minimize(loss_cross_entropy)
# define accuracy
correct_prediction = tf.equal(tf.argmax(pedprob,1), tf.argmax(true_out,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# sess.run(tf.initialize_all_variables())
tfsaver = tf.train.Saver()
tfsaver.restore(sess, "./PARSE224-VGG19_2.ckpt")
print("restoration complete...")
BS = BATCH_SIZE



'''
for i in range(MAX_STEPS):
	#batch = read_train('../PARSE224', BS, NUM_LABELS)	
	#print batch[0].shape
	#print batch[1].shape

	print 'iter', i
	# print a log
	if (i%5000 == 0):
		#train_accuracy = accuracy.eval(feed_dict={images: batch[0], true_out: batch[1], train_mode: False})
		print("step %d, training accuracy %g"%(i, 100.0*train_accuracy))
		
		batch2 = read_valid('../PARSE224', NUM_LABELS)
		
		it = 0
		test_accuracy2 = 0
		while it*BS<6500:
			test_accuracy2 = (it*test_accuracy2+accuracy.eval(feed_dict={images: batch2[0][it*BS:((it+1)*BS),:], true_out: batch2[1][it*BS:((it+1)*BS),:], train_mode:False}))/(it+1)
			it = it + 1
		print("step %d, test accuracy %g"%(i, 100.0*test_accuracy2))
		


		tmpstr = ("./conv3/../PARSE224-VGG19_2.ckpt")
		print tmpstr
		#save_path = tfsaver.save(sess, tmpstr)

	# run train step
	#train_step.run(feed_dict={images: batch[0], true_out: batch[1], train_mode:True})
'''

datadir = '/home/ashish/Desktop/ped-det/data/ActivityDataset'

for i in range(44):
	seqn = i+1;
	seqdir = datadir + '/seq%2.2d'%(seqn)
	print 'reading images from ' + seqdir
	# os.chdir(seqdir)
	# images = [file for file in glob.glob("*.jpg")]
	# for image in images:
	# 	print image
	with open(seqdir+'/tracks.csv', 'rb') as f:
	    reader = csv.reader(f)
	    tracks = list(reader)
	annot = []
	for track in tracks:
		frame = track[0]
		im = Image.open(seqdir+'/'+'frame%4.4d.jpg'%(int(frame)))
		x = int(float(track[2]))
		y = int(float(track[3]))
		w = int(float(track[4]))
		h = int(float(track[5]))
		im = numpy.array(im)
		up = max(0,-im.shape[0]+y+h+1)
		down = max(0,0-y)
		right = max(0,-im.shape[1]+x+w+1)
		left = max(0,0-x)
		x = x+left
		y = y+up
		# print im.shape, x, x+w, y, y+h
		im = numpy.lib.pad(im, ((up,down),(left,right),(0,0)), 'edge')
		im_ = misc.imresize(im[y:y+h,x:x+w,:],(128,64))
		im__ = ((misc.imresize(im_, (224,224))).astype(float32))/255
		probs = pedprob.eval(feed_dict={images: [im__], train_mode: False})
		probs = probs.squeeze().tolist()
		# res =  ["%.2f" % v for v in probs]
		# res_str = 'res: '
		# for a_string in res:
		# 	res_str += a_string + ' | '
		# plt.imshow(im_)
		print ('seuence: %d, frame: %d, person: %d, %d of %d processed'%(seqn, int(frame), int(track[1]), len(annot), len(tracks)))
		# plt.title(res_str)
		# plt.pause(0.5)
		prop = [int(track[0]), int(track[1]), x,y,w,h,]
		prop += probs
		annot.append(prop)

	with open(seqdir + "/orientations.csv", "wb") as f:
	    writer = csv.writer(f)
	    writer.writerows(annot)

			


	# os.chdir(datadir)
	

	

"""
it = 0
test_accuracy2 = 0
while it*BS<6500:
	batch2 = read_valid('../PARSE224', NUM_LABELS, it*BS,it*BS+BS)
	test_accuracy2 = (it*test_accuracy2+accuracy.eval(feed_dict={images: batch2[0], true_out: batch2[1], train_mode:False}))/(it+1)
	it = it + 1
	print("validation accuracy %g"%(100.0*test_accuracy2))

"""

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
