data_dir = './ActivityDataset';
testseq = [30    32    33    42    44    18    31    29    34    14    27];
anno_dir = fullfile(data_dir, 'Final_annotations');
Theta = [0, 45, 90, 135, 180, 225, 270, 315]*pi/180;

trainX1 = [];
trainX2 = [];
trainY = [];
testX1 = [];
testX2 = [];
testY = [];

%%
for seqi = 1:44
    if any(testseq == seqi), continue, end
    if seqi == 39, continue, end
    annofile = fullfile(anno_dir, sprintf('data_%2.2d.mat', seqi));
    display(annofile)
    anno = load(annofile);
    anno = anno.anno_data;
    t = 31;
    resnet50 = csvread(fullfile(data_dir, sprintf('seq%2.2d', seqi), 'resnet50.txt'));
    while(t < anno.nframe)
        feat_vec = [];
        label = [];
        nped = 0;
        res_vec = resnet50(t-9:t,:);
        for ped = anno.people
            bbs = ped.bbs(t-9:t, :);
            actions = ped.action(t-9:t);
            bbs(:,1) = bbs(:,1) + bbs(:,3)/2;
            bbs(:,2) = bbs(:,2) + bbs(:,4)/1;
            bbf = ped.bbs(t-29:t, :);
            bbf(:,1) = bbf(:,1) + bbf(:,3)/2;
            bbf(:,2) = bbf(:,2) + bbf(:,4)/1;
            validt = sum(bbf, 2)>0;
            if sum(validt) < 10, continue, end
            tvec = 1:30;
            tvec = tvec(validt);
            bias = ones(size(tvec));
            predictor = [bias', tvec'];
            [Bx, ~, ~] = regress(bbf(validt,1), predictor);
            [By, ~, ~] = regress(bbf(validt,2), predictor);
            [Bw, ~, ~] = regress(bbf(validt,3), predictor);
            [Bh, ~, ~] = regress(bbf(validt,4), predictor);
            Bx = Bx(2) * ones(size(actions));
            By = By(2) * ones(size(actions));
            Bw = Bw(2) * ones(size(actions));
            Bh = Bh(2) * ones(size(actions));
            poses = ped.pose(t-9:t);
            poses_debug = poses;
            poses_debug(poses == 0) = 1;
            thetas = Theta(poses_debug);
            cosines = cos(thetas).*double(poses ~= 0);
            sines = sin(thetas).*double(poses ~= 0);

            fvec = [bbs, cosines', sines', actions'-1, Bx', By', Bw', Bh'];

            if numel(feat_vec) == 0
                feat_vec = fvec;
                nped = 1;
            else
                feat_vec = cat(2, feat_vec, fvec);
                nped = nped + 1;
            end
        end
        if nped < 1
            display('skipped this scene')
        elseif nped < 20
            npad = 20-nped;
            padsize = [10, npad*11];
            feat_vec = cat(2, feat_vec, zeros(padsize));
            if numel(trainX1) == 0
                trainX1 = reshape(feat_vec, [1, 10, 20*11]);
            else
                trainX1(end+1, :, :) = feat_vec;
            end
            
        else
            feat_vec = feat_vec(:, 1:20*11);
            if numel(trainX1) == 0
                trainX1 = reshape(feat_vec, [1, 10, 20*11]);
            else
                trainX1(end+1, :, :) = feat_vec;
            end
            
        end
        if nped > 0
            trainX2(end+1, :, :) = res_vec;
            trainY(end+1, 1) = anno.Collective(t);
        end
        t = t + 10;
    end
end

%%
for seqi = 1:44
    if ~any(testseq == seqi), continue, end
    if seqi == 39, continue, end
    annofile = fullfile(anno_dir, sprintf('data_%2.2d.mat', seqi));
    display(annofile)
    anno = load(annofile);
    anno = anno.anno_data;
    t = 31;
    resnet50 = csvread(fullfile(data_dir, sprintf('seq%2.2d', seqi), 'resnet50.txt'));
    while(t < anno.nframe)
        feat_vec = [];
        label = [];
        nped = 0;
        res_vec = resnet50(t-9:t,:);
        for ped = anno.people
            bbs = ped.bbs(t-9:t, :);
            actions = ped.action(t-9:t);
            bbs(:,1) = bbs(:,1) + bbs(:,3)/2;
            bbs(:,2) = bbs(:,2) + bbs(:,4)/1;
            bbf = ped.bbs(t-29:t, :);
            bbf(:,1) = bbf(:,1) + bbf(:,3)/2;
            bbf(:,2) = bbf(:,2) + bbf(:,4)/1;
            validt = sum(bbf, 2)>0;
            if sum(validt) < 10, continue, end
            tvec = 1:30;
            tvec = tvec(validt);
            bias = ones(size(tvec));
            predictor = [bias', tvec'];
            [Bx, ~, ~] = regress(bbf(validt,1), predictor);
            [By, ~, ~] = regress(bbf(validt,2), predictor);
            [Bw, ~, ~] = regress(bbf(validt,3), predictor);
            [Bh, ~, ~] = regress(bbf(validt,4), predictor);
            Bx = Bx(2) * ones(size(actions));
            By = By(2) * ones(size(actions));
            Bw = Bw(2) * ones(size(actions));
            Bh = Bh(2) * ones(size(actions));
            poses = ped.pose(t-9:t);
            poses_debug = poses;
            poses_debug(poses == 0) = 1;
            thetas = Theta(poses_debug);
            cosines = cos(thetas).*double(poses ~= 0);
            sines = sin(thetas).*double(poses ~= 0);

            fvec = [bbs, cosines', sines', actions'-1, Bx', By', Bw', Bh'];

            if numel(feat_vec) == 0
                feat_vec = fvec;
                nped = 1;
            else
                feat_vec = cat(2, feat_vec, fvec);
                nped = nped + 1;
            end
        end
        if nped < 1
            display('skipped this scene')
        elseif nped < 20
            npad = 20-nped;
            padsize = [10, npad*11];
            feat_vec = cat(2, feat_vec, zeros(padsize));
            if numel(testX1) == 0
                testX1 = reshape(feat_vec, [1, 10, 20*11]);
            else
                testX1(end+1, :, :) = feat_vec;
            end
            
        else
            feat_vec = feat_vec(:, 1:20*11);
            if numel(testX1) == 0
                testX1 = reshape(feat_vec, [1, 10, 20*11]);
            else
                testX1(end+1, :, :) = feat_vec;
            end
            
        end
        if nped > 0
            testX2(end+1, :, :) = res_vec;
            testY(end+1, 1) = anno.Collective(t);
        end
        t = t + 10;
    end
end

%%
trainX1 = reshape(trainX1, [size(trainX1, 1), size(trainX1, 2)*size(trainX1, 3)]);
testX1 = reshape(testX1, [size(testX1, 1), size(testX1, 2)*size(testX1, 3)]);
trainX2 = reshape(trainX2, [size(trainX2, 1), size(trainX2, 2)*size(trainX2, 3)]);
testX2 = reshape(testX2, [size(testX2, 1), size(testX2, 2)*size(testX2, 3)]);

outdir = './ind2scene/';
mkdir(outdir)
csvwrite(fullfile(outdir, 'trainX1.csv'), trainX1);
csvwrite(fullfile(outdir, 'trainX2.csv'), trainX2);
csvwrite(fullfile(outdir, 'trainY.csv'), trainY);
csvwrite(fullfile(outdir, 'testX1.csv'), testX1);
csvwrite(fullfile(outdir, 'testX2.csv'), testX2);
csvwrite(fullfile(outdir, 'testY.csv'), testY);