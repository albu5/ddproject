%%
% function deptest(imgdir, detfile, minsize, maxsize, thresh)
% imgdir = 'data/temple/';
% detfile = 'det.txt';
addpath(genpath('Helpers'))

if ~exist('thresh', 'var')
    thresh = -0.05;
end

files = dir([imgdir filesep '*.jp*']);

im0 = fullfile(imgdir, files(1).name);
im1 = fullfile(imgdir, files(2).name);

res = demo1(im0, im1, minsize, maxsize);
resfilt = res(res(:,5)>thresh,:);
seq_det = [1*ones(size(resfilt,1), 1), -1*ones(size(resfilt,1), 1), resfilt, -1*ones(size(resfilt,1), 3)];
figure
set(gcf, 'Position', [0 0 size(imread(im0), 1), size(imread(im0),2)])
for i = 1:numel(files)-1
    im0 = fullfile(imgdir, files(i).name);
    im1 = fullfile(imgdir, files(i+1).name);
    res = demo1(im0, im1, minsize, maxsize);
    resfilt = res(res(:,5)>thresh,:);
    seq_det = vertcat(seq_det, [(i+1)*ones(size(resfilt,1), 1), -1*ones(size(resfilt,1), 1), resfilt, -1*ones(size(resfilt,1), 3)]);
    I = insertObjectAnnotation(imread(im0), 'rectangle', round(resfilt(:,1:4)), resfilt(:,5), 'TextBoxOpacity',0.9);
    step(v, I);
end
close
%%
fid = fopen(detfile, 'w');
fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\r\n', seq_det');
fclose(fid);
