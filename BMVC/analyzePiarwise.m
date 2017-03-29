trainidx = rand(size(Table,1),1)>0.4;

TableTrain = Table(trainidx, :);
TableTest = Table(~trainidx, :);

trainX = TableTrain(:,[1,2,4,5,6,7,9,10,12,13,14,15]);
testX = TableTest(:,[1,2,4,5,6,7,9,10,12,13,14,15]);
trainY = TableTrain(:,8) == TableTrain(:,16);
testY = TableTest(:,8) == TableTest(:,16);
mydata = array2table(trainX);
mydata.classes = categorical(trainY);

%%
table2 = mydata;
table2(1:numel(testY),1:12) = array2table(testX);
table2(numel(testY)+1:end,:) = [];
table2.classes = [];
