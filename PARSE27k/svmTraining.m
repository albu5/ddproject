Xtrain = csvread('PARSE27/train/hog_vec.txt');
Ytrain = csvread('PARSE27/train/labels.txt');
%%
Xvalid = csvread('PARSE27/valid/hog_vec.txt');
Yvalid = csvread('PARSE27/valid/labels.txt');
%%
mu = mean(Xtrain,1);
sigma = std(Xtrain,1);
%%
Xtrain = (Xtrain-repmat(mu,[size(Xtrain,1),1]));
Xvalid = (Xvalid-repmat(mu,[size(Xvalid,1),1]));
Xtrain = Xtrain./repmat(sigma,[size(Xtrain,1),1]);
Xvalid = Xvalid./repmat(sigma,[size(Xvalid,1),1]);

%%
mysvm = templateSVM('Standardize', 1, 'Verbose', 1);
Model1 = fitcecoc(Xtrain,categorical(Ytrain),'Learners', mysvm);

%%
trainp = double(predict(Model1, Xtrain));
%%
mean(trainp==Ytrain)
%%
validp = double(predict(Model1, Xvalid));
%%
mean(validp==Yvalid)