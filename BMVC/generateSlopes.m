mkdir('./ActivityDataset/trajSlopes')
csvwrite('./ActivityDataset/trajSlopes/train_A', horzcat(trainX(:,1:12), trainY));
csvwrite('./ActivityDataset/trajSlopes/test_A', horzcat(testX(:,1:12), testY));
