%%
function success = deptest(imgdir, detfile, minsize, maxsize, thresh, disp, ext)
% imgdir = 'data/temple/';
% detfile = 'det.txt';

success = 0;

addpath(genpath('Helpers'))

if ~exist('thresh', 'var')
    thresh = -0.5;
end

if ~exist('disp', 'var')
    disp = false;
end

if ~exist('ext', 'var')
    ext = 'jp';
end


files = dir([imgdir filesep '*.' ext '*']);

im0 = fullfile(imgdir, files(1).name);
im1 = fullfile(imgdir, files(2).name);

res = demo1(im0, im1, minsize, maxsize);
resfilt = res(res(:,5)>thresh,:);
seq_det = [];

if disp
    figure
    set(gcf, 'Position', [0 0 size(imread(im0), 1), size(imread(im0),2)])
end

fnames1 = [];
fnames2 = [];

seq_det(1).resfilt = [1*ones(size(resfilt,1), 1), -1*ones(size(resfilt,1), 1), resfilt, -1*ones(size(resfilt,1), 3)];

for i = 1:numel(files)-1
    fnames1{i} = files(i).name;
    fnames2{i} = files(i+1).name;
    seq_det(i+1).resfilt = [];
end


for i = 1:numel(files)-1
    im0 = fullfile(imgdir, fnames1{i});
    im1 = fullfile(imgdir, fnames2{i});
    res = demo1(im0, im1, minsize, maxsize);
    resfilt = res(res(:,5)>thresh,:);
    seq_det(i+1).resfilt = [(i+1)*ones(size(resfilt,1), 1), -1*ones(size(resfilt,1), 1), resfilt, -1*ones(size(resfilt,1), 3)];
    if disp
        I = insertObjectAnnotation(imread(im0), 'rectangle', round(resfilt(:,1:4)), 'Person', 'TextBoxOpacity',0.9);
        imagesc(I), axis image, title(sprintf('%d out of %d frames', i, numel(files)));
        pause(0.001)
    end
end
close
%%
seq_det_vec = [];
for i = 1:numel(seq_det)
    seq_det_vec = vertcat(seq_det_vec, seq_det(i).resfilt);
end
fid = fopen(detfile, 'w');
fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\r\n', seq_det_vec');
fclose(fid);

success = 1;
