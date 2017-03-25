% Parsing data for classification learner
% 
% inputs = 
%   image foldername with subfolder location
%   label filename
% output = 
%   train which is a categorical table
% 
% creates table (including array2table & categorical) needed for training

function train = createHOGTraining(foldername)

trainx = [];

% formatSpec = strcat(foldername,'\%d.png');
fdne = 0;
i = 1;

while fdne==0
    try
    im = imread(fullfile(foldername,sprintf('%d.png',i)));
    im_hog = extractHOGFeatures(im);
    trainx = vertcat(trainx,im_hog);
    i = i+1;
    catch
        break
    end
%     if exist(sprintf(formatSpec,i+1),'file')
%         i = i+1;
%     else
%         fdne = 1;
%         sprintf('All images converted!');
%     end
end

label = strcat(foldername,'/labels.txt');

if exist(label,'file')
    trainy = csvread(label);
    
    train = array2table(trainx);
    train.class = categorical(trainy);
    
else
    sprintf('Label file does not exist!');
end

    