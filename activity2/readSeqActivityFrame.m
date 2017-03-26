%%
datadir = '../../data/ActivityDataset';
sequences = dir([datadir, filesep, 'seq*']);
outdir = [datadir filesep 'individuals_frame_10'];
tstep = 10;
%%
if(~exist(outdir, 'dir'))
    mkdir(outdir)
    mkdir(outdir,'train')
    mkdir(outdir,'valid')
end

%%idx
testseq = [1     4     5     6     8     2     7    28    35    11    10    26];
imc = 1;

for seqn = 1:44
    labels = [];
    curr_seq = (fullfile(outdir, 'train', sprintf('seq%2.2d',seqn)));
    display(curr_seq)
    mkdir(curr_seq)
    if (sum(testseq==seqn)>0), continue, end
%     close all

    load(fullfile(datadir, 'annotations', sprintf('anno%2.2d.mat', seqn)));
    pedn = 0;
    for ped = anno.people
        pedn = pedn + 1;
        display(sprintf('sequence: %d, pedestrian: %d', seqn, pedn))
        t = ped.time;
        bbs = ped.sbbs;
        attr = ped.attr;
        i = 1;
        while i<numel(t)
                im = imread(fullfile(datadir, sprintf('seq%2.2d', seqn), sprintf('frame%4.4d.jpg', t(i))));
                im_ = imcrop(im, [bbs(1,i), bbs(2,i), bbs(3,i), bbs(4,i)]);
                if numel(im_) == 0, i=i+1;continue; end
                curr_attr = attr(2,i);
                im_ = imresize(im_, [224, 112]);
                i = i + (attr(2,i)^2)*tstep;
                imagesc(im_), axis image
                inp = (input(sprintf('Current activity: %d, Label:',curr_attr)));
                if inp>0
                labels(imc) = inp;
                imwrite(im_, fullfile(outdir, 'train', sprintf('seq%2.2d',seqn), sprintf('%d.jpg',imc)));
                imc = imc + 1;
                end
        end
        
    end
csvwrite(fullfile(outdir,'train',sprintf('seq%2.2d',seqn), 'labels.txt'), labels(:));
end
%%
pause
%%
testseq = [1     4     5     6     8     2     7    28    35    11    10    26];
imc = 1;
labels = [];
for seqn = 1:44
    if (sum(testseq==seqn)==0), continue, end
    close all
    load(fullfile(datadir, 'annotations', sprintf('anno%2.2d.mat', seqn)));
    pedn = 0;
    for ped = anno.people
        pedn = pedn + 1;
        display(sprintf('sequence: %d, pedestrian: %d', seqn, pedn))
        t = ped.time;
        bbs = ped.sbbs;
        attr = ped.attr;
        i = 1;
        while i<numel(t)
                im = imread(fullfile(datadir, sprintf('seq%2.2d', seqn), sprintf('frame%4.4d.jpg', t(i))));
                im_ = imcrop(im, [bbs(1,i), bbs(2,i), bbs(3,i), bbs(4,i)]);
                if numel(im_) == 0, i=i+tstep;continue; end
                im_ = imresize(im_, [128, 64]);
                imwrite(im_, fullfile(outdir, 'valid', sprintf('%d.jpg',imc)));
                imc = imc + 1;
                i = i + tstep;
                labels(imc-1) = attr(2,i);
        end
        
    end
end
csvwrite(fullfile(outdir,'valid','labels.txt'), labels(:));