function result = testHOGClassifier(foldername,classifier)
% test the trained classifier on the test images
% of same format as training images
% uses classifier to test and validate the results and gives predictions 
% (0 for wrong prediction if pre-annotated)

testx = [];
testy = [];
formatSpec = strcat(foldername,'/%d.png');
fdne = 0;
i = 1;

while fdne==0
    im = imread(sprintf(formatSpec,i));
    im_hog = extractHOGFeatures(im);
    
    prediction = predict(classifier, im_hog);
    testx = vertcat(testx,im_hog);
    testy = vertcat(testy,prediction);
    
    if exist(sprintf(formatSpec,i+1),'file')
        i = i+1;
    else
        fdne = 1;
    end
    
end

% compare results from label
label = strcat(foldername,'/labels.txt');
if exist(label,'file')
    validy = csvread(label);
    testy = double(testy);
    result = testy.*((testy-validy)<0.5);  % 0/1 for correct(in) classification
else
    sprintf('Label file does not exist!');
    result = testy;
end