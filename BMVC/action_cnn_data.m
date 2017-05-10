%%
datadir = './ActivityDataset';
outdir = [datadir filesep 'ActionCNN'];
annodir = fullfile(datadir, 'my_anno');
%%
if(~exist(outdir, 'dir'))
    mkdir(outdir)
end
testseq = [1     4     5     6     8     2     7    28    35    11    10    26];
%%
trainX = zeros(12000, 224*224*3);
trainY = zeros(12000, 1);
train = 1;
for seqi = 1:44
    if any(testseq == seqi), continue, end
    seq_dir = fullfile(datadir, sprintf('seq%2.2d', seqi));
    display(seq_dir)
    train
    annopath = fullfile(annodir, sprintf('data_%2.2d.mat', seqi));
    anno = load(annopath);
    anno = anno.anno_data;
    for t = 1:10:anno.nframe
        im = imread(fullfile(seq_dir, sprintf('frame%4.4d.jpg', t)));
        % %         imshow(im)
        for person = anno.people
            bb =  person.bbs(t,:);
            im_ = imcrop(im, bb);
            if numel(im_) == 0, continue, end
            %             imshow(im_)
            im__ = imresize(double(im_), [224, 224]);
            trainX(train, :) = im__(:)';
            trainY(train, 1) = person.action(t);
            train = train + 1;
            if (trainY(end) == 0), trainY(end) = 1; end
        end
    end
    
end

%%
testX = zeros(12000, 224*224*3);
testY = zeros(12000, 1);
test = 1;
for seqi = 1:44
    if ~any(testseq == seqi), continue, end
    seq_dir = fullfile(datadir, sprintf('seq%2.2d', seqi));
    display(seq_dir)
    test
    annopath = fullfile(annodir, sprintf('data_%2.2d.mat', seqi));
    anno = load(annopath);
    anno = anno.anno_data;
    for t = 1:10:anno.nframe
        im = imread(fullfile(seq_dir, sprintf('frame%4.4d.jpg', t)));
        %         imshow(im)
        for person = anno.people
            bb =  person.bbs(t,:);
            im_ = imcrop(im, bb);
            if numel(im_) == 0, continue, end
            %             imshow(im_)
            im__ = imresize(double(im_), [224, 224]);
            
            testX(test, :) = im__(:)';
            testY(test, 1) = person.action(t);
            test = test + 1;
            if (testY(end) == 0), testY(end) = 1; end
        end
    end
    
end

%%
%%
trainX(train:end, :) = [];
trainY(train:end, :) = [];
testX(test:end, :) = [];
testY(test:end, :) = [];

%%
hdf5write('action_cnn.h5', '/trainX', trainX);
hdf5write('action_cnn.h5', '/trainY', trainY, 'WriteMode', 'append');
hdf5write('action_cnn.h5', '/testX', testX, 'WriteMode', 'append');
hdf5write('action_cnn.h5', '/testY', testY, 'WriteMode', 'append');