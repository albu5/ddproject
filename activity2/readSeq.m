%%
datadir = '../../data/ActivityDataset';
sequences = dir([datadir, filesep, 'seq*']);
thresh = 29;
outdir = [datadir filesep 'individuals30_split1_tracks'];

%%
if(~exist(outdir, 'dir'))
    mkdir(outdir)
    mkdir(outdir,'train')
    mkdir(fullfile(outdir,'train','1'))
    mkdir(fullfile(outdir,'train','2'))
    mkdir(outdir,'valid')
    mkdir(fullfile(outdir,'valid','1'))
    mkdir(fullfile(outdir,'valid','2'))
end

pause
%%
counter1 = 1;
counter2 = 1;

testseq = [1     4     5     6     8     2     7    28    35    11    10    26];

for idx = 1:44
    if (sum(testseq==idx)>0), continue, end
    close all
    load(fullfile(datadir, 'annotations', sprintf('anno%2.2d.mat', idx)));
    for ped = anno.people
        t = ped.time;
        bbs = ped.sbbs;
        attr = ped.attr;
        i = 1;
        while i<numel(t)
            curr_attr = attr(2,i);
            display(sprintf('scene: %d activity: %d', idx, curr_attr));
            if curr_attr == 1
                counter = counter1;
            elseif curr_attr == 2
                counter = counter2;
            else
                i = i+1;
                continue;
            end
            imc = 1;
            mkdir(fullfile(outdir, 'train', num2str(curr_attr), ...
                num2str(counter)));
            mydata = [];
            while (curr_attr == attr(2,i)) && (i<numel(t)) && (imc<thresh+2)
                im = imread(fullfile(datadir, sprintf('seq%2.2d', idx), sprintf('frame%4.4d.jpg', t(i))));
                im_ = imcrop(im, [bbs(1,i), bbs(2,i), bbs(3,i), bbs(4,i)]);
                if numel(im_) == 0, i=i+1;continue; end
                im_ = imresize(im_, [128, 64]);
                [fvec,~] = extractHOGFeatures(im_);
                fvec = horzcat(fvec, bbs(1:4,i)');
                mydata = vertcat(mydata,fvec);
%                 imagesc(imresize(im_,1)), axis image
                %                 [~,vis] = extractHOGFeatures(im_); hold on, plot(vis), hold off
                
                display(sprintf('scene: %d activity: %d', idx, curr_attr));
                %                 title(sprintf('scene: %d activity: %d', idx, curr_attr));
%                 imwrite(im_, fullfile(outdir, 'train', num2str(curr_attr), ...
%                     num2str(counter), sprintf('%d.jpg', imc)));
                imc = imc + 1;
                pause(0.000001)
                i=i+1;
            end
            csvwrite(fullfile(outdir, 'train', num2str(curr_attr),num2str(counter),'hog_track.csv'),mydata);
            if curr_attr == 1
                counter1 = counter1+1;
            else
                counter2 = counter2+1;
            end
        end
        
    end
end

%%

%
counter1 = 1;
counter2 = 1;

testseq = [1     4     5     6     8     2     7    28    35    11    10    26];

for idx = 1:44
    if (sum(testseq==idx)==0), continue, end
    close all
    load(fullfile(datadir, 'annotations', sprintf('anno%2.2d.mat', idx)));
    for ped = anno.people
        t = ped.time;
        bbs = ped.sbbs;
        attr = ped.attr;
        i = 1;
        while i<numel(t)
            curr_attr = attr(2,i);
            display(sprintf('scene: %d activity: %d', idx, curr_attr));
            if curr_attr == 1
                counter = counter1;
            elseif curr_attr == 2
                counter = counter2;
            else
                i = i+1;
                continue;
            end
            imc = 1;
            mkdir(fullfile(outdir, 'valid', num2str(curr_attr), ...
                num2str(counter)));
            mydata = [];
            while (curr_attr == attr(2,i)) && (i<numel(t)) && (imc<thresh+2)
                im = imread(fullfile(datadir, sprintf('seq%2.2d', idx), sprintf('frame%4.4d.jpg', t(i))));
                im_ = imcrop(im, [bbs(1,i), bbs(2,i), bbs(3,i), bbs(4,i)]);
                if numel(im_) == 0, i=i+1;continue; end
                im_ = imresize(im_, [128, 64]);
                [fvec,~] = extractHOGFeatures(im_);
                fvec = horzcat(fvec, bbs(1:4,i)');
                mydata = vertcat(mydata,fvec);
%                 imagesc(imresize(im_,1)), axis image
                %                 [~,vis] = extractHOGFeatures(im_); hold on, plot(vis), hold off
                
                display(sprintf('scene: %d activity: %d', idx, curr_attr));
                %                 title(sprintf('scene: %d activity: %d', idx, curr_attr));
%                 imwrite(im_, fullfile(outdir, 'valid', num2str(curr_attr), ...
%                     num2str(counter), sprintf('%d.jpg', imc)));
                imc = imc + 1;
                pause(0.000001)
                i=i+1;
            end
            csvwrite(fullfile(outdir, 'valid', num2str(curr_attr),num2str(counter),'hog_track.csv'),mydata);
            if curr_attr == 1
                counter1 = counter1+1;
            else
                counter2 = counter2+1;
            end
        end
        
    end
end

%%
% %%
% showTracks(fullfile(datadir, [sequences(1).name, '/']), tracks)
%
% %%
% showDets(fullfile(datadir, [sequences(1).name, '/']), dets)
