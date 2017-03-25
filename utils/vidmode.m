clear
imgdir = '../../data/eebuilding2/img';
files = dir([imgdir filesep '*.jp*']);

M = 100;

im1 = imread(fullfile(imgdir, files(1).name));
viddata = zeros([numel(im1), 256]);
for i = 1:numel(files)
    im0 = imread(fullfile(imgdir, files(i).name));
    im0 = im0(:);
    for j = 1:numel(im0)
        viddata(j,im0(j)+1) = viddata(j,im0(j)+1) + 1;
    end
    clc
    display(num2str(100*i/min(M,numel(files))));
    if i>M
        break
    end
end
%%

[~, vidmode] = max(viddata, [], 2);
vidmode = reshape(vidmode, size(im1));
imagesc(uint8(vidmode))
imwrite(uint8(vidmode), [imgdir '/../vidmode.jpg']);