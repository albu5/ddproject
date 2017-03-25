import tflearn
from readActivity import Dataset

# Parameters
learning_rate = 0.001
training_iters = 1000000
batch_size = 32
display_step = 1
valid_step = 20
decay_step = 50
fveclen = 3780

# Network Parameters
n_hidden = 128 # hidden layer num of features
n_classes = 2 # linear sequence or not

feat = 'hog.csv'
datadir = '/home/ashish/Desktop/ped-det/data/ActivityDataset/individuals_short'
max_seqlen = 91
split_tol = 0.01

dataset = Dataset(feat=feat, datadir=datadir, max_seqlen=max_seqlen, split_tol=split_tol)
trainX, trainY, trainLen = dataset.getTrainBatch(1223)
validX, validY, validLen = dataset.getValidBatch(612)

input_data = tflearn.input_data([None, max_seqlen])
lstm = tflearn.lstm(input_data, n_hidden, dropout=0.8, dynamic=True)
pred = tflearn.fully_connected(lstm, 2, activation='softmax')
net = tflearn.regression(pred, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=3)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=batch_size)