load multiview_traindata.mat;
addpath(genpath('..\..\..\external\dbn-toolbox'))

%%
train_x = double(reshape(trainx', [100 40 size(trainx,1)]))/255;
train_y = double(full(ind2vec(trainy')));
%%
% train_x = double(reshape(trainx',28,28,60000))/255;
% test_x = double(reshape(test_x',28,28,10000))/255;
% train_y = double(train_y');
% test_y = double(test_y');

%% ex1 Train a 6c-2s-12c-2s Convolutional neural network
%will run 1 epoch in about 200 second and get around 11% error.
%With 100 epochs you'll get around 1.2% error

rand('state',0)

cnn.layers = {
    struct('type', 'i') %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
    struct('type', 's', 'scale', 2) %subsampling layer
    };


opts.alpha = 1;
opts.batchsize = 52;
opts.numepochs = 200;
opts.plot = true;

cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);

% [er, bad] = cnntest(cnn, test_x, test_y);

%plot mean squared error
figure; plot(cnn.rL);
% assert(er<0.12, 'Too big error');

%% validation
load multiview_validdata.mat;
valid_x = double(reshape(validx', [100 40 size(validx,1)]))/255;
valid_y = double(full(ind2vec(validy')));
[er, bad] = cnntest(cnn, test_x, test_y);