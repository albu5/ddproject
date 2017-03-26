trainD = array2table(trainX(:,1:12));
trainD.classes = categorical(trainY);

testD = array2table(testX(:,1:12));

% testD(1:size(testX,1),1:size(testX,2)) = array2table(testX);
% testD = testD(1:size(testX,1),1:size(testX,2));
