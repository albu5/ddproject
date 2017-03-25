imgdir = '../../data/temple5/img';
files = dir([imgdir filesep '*.jp*']);

vidmode = double(imread([imgdir '/../vidmode.jpg']));


bwthresh = 10;

for i = 1:numel(files)
    im0 = double(imread(fullfile(imgdir, files(i).name)));
    imdiff = mean(abs(im0-vidmode), 3);
    
    sel = strel('disk', 5);
    bw1 = imdiff>bwthresh;
    bw2 = imerode(bw1, sel);
    bw3 = imdilate(bw2, sel);
    bw4 = bwareaopen(bw3, 400);
    imagesc(bw4), axis image
    pause(1/30);
end
