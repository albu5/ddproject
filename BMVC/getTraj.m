%%
datadir = './ActivityDataset';
sequences = dir([datadir, filesep, 'seq*']);
delt = inf;
outdir = [datadir filesep 'TrajectoriesLong'];

%%
if(~exist(outdir, 'dir'))
    mkdir(outdir)
end

%%
trainTrack = {};
trainPose = {};
trainAction = {};


testseq = [1     4     5     6     8     2     7    28    35    11    10    26];
nex = 1;
for idx = 1:44
    if (sum(testseq==idx)>0), continue, end
    load(fullfile(datadir, 'my_anno', sprintf('data_%2.2d.mat', idx)));
    seqdir = fullfile(datadir, sprintf('seq%2.2d',idx));
    display(num2str(idx))
    nped = 0;
    for ped = anno_data.people
        nped = nped + 1;
        bbs = ped.bbs;
        validt = sum(bbs,2)>0;
        start = 1;
        curr = 1;
        tracklet = [];
        actionlet = [];
        while(curr<numel(validt))
            while true
                curr = curr + 1;
                if ~validt(curr)
                    if curr-start > 1
                        exdir = fullfile(outdir, 'train', num2str(nex));
                        mkdir(exdir)
                        bf = ped.bbs(start:curr-1,:);
                        pf = ped.pose(start:curr-1);
                        af = ped.action(start:curr-1);
                        linind = start:curr-1;
                        for t = start:curr-1
                            rect = ped.bbs(t,:);
                            imgname = fullfile(seqdir, sprintf('frame%4.4d.jpg',t));
                            im = imread(imgname);
                            im_ = imcrop(im, rect);
                            try
                                im__ = imresize(im_, [128,NaN]);
                                imwrite(im__, fullfile(exdir, sprintf('%4.4d.jpg',t)));
                            catch
                                bf(t-start+1,:) = [];
                                pf(t-start+1) = [];
                                af(t-start+1) = [];
                                linind(t-start+1) = [];
                            end
                            %                             imshow(im__), drawnow;
                        end
                        csvwrite(fullfile(exdir, 'tracks.txt'), [linind; bf'; af; pf]);
                        trainTrack{end+1} = bf;
                        trainPose{end+1} = pf;
                        trainAction{end+1} = af;
                        nex = nex+1;
                    end
                    start = curr;
                    break;
                end
            end
        end
    end
end

%%
testTrack = {};
testPose = {};
testAction = {};


testseq = [1     4     5     6     8     2     7    28    35    11    10    26];
nex = 1;
for idx = 1:44
    if (sum(testseq==idx)==0), continue, end
    load(fullfile(datadir, 'my_anno', sprintf('data_%2.2d.mat', idx)));
    seqdir = fullfile(datadir, sprintf('seq%2.2d',idx));
    display(num2str(idx))
    nped = 0;
    for ped = anno_data.people
        nped = nped + 1;
        bbs = ped.bbs;
        validt = sum(bbs,2)>0;
        start = 1;
        curr = 1;
        tracklet = [];
        actionlet = [];
        while(curr<numel(validt))
            while true
                curr = curr + 1;
                if ~validt(curr)
                    if curr-start > 1
                        exdir = fullfile(outdir, 'test', num2str(nex));
                        mkdir(exdir)
                        bf = ped.bbs(start:curr-1,:);
                        pf = ped.pose(start:curr-1);
                        af = ped.action(start:curr-1);
                        linind = start:curr-1;
                        for t = start:curr-1
                            rect = ped.bbs(t,:);
                            imgname = fullfile(seqdir, sprintf('frame%4.4d.jpg',t));
                            im = imread(imgname);
                            im_ = imcrop(im, rect);
                            try
                                im__ = imresize(im_, [128,NaN]);
                                imwrite(im__, fullfile(exdir, sprintf('%4.4d.jpg',t)));
                            catch
                                bf(t-start+1,:) = [];
                                pf(t-start+1) = [];
                                af(t-start+1) = [];
                                linind(t-start+1) = [];
                            end
                            %                             imshow(im__), drawnow;
                        end
                        csvwrite(fullfile(exdir, 'tracks.txt'), [linind; bf'; af; pf]);
                        testTrack{end+1} = bf;
                        testPose{end+1} = pf;
                        testAction{end+1} = af;
                        nex = nex+1;
                    end
                    start = curr;
                    break;
                end
            end
        end
    end
end
