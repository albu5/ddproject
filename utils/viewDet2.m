% function viewDet(imgdir, trackfile)
datadir = '../../data/eebuilding2';

imgdir = [datadir filesep 'img'];
trackfile = [datadir filesep 'det.txt'];
detfile = [datadir filesep 'detfilt.txt'];
vidmodefile = [datadir filesep 'vidmode.jpg'];
vidmode = imread(vidmodefile);

addpath(genpath('../../external/toolbox-master'))
files = dir([imgdir filesep '*.jp*']);
v = vision.VideoPlayer;
v.Position = [1 1 1920 1080];
tracks = csvread(trackfile);
c = ceil(256*rand(512,3));
cdata = [];

fthresh = 0.1;
overlapthresh = 0.35;
scorethresh = -0.2;
minsize = 210;
maxsize = 500;

opts.bwthresh = 10;
opts.strel = 4;
opts.minSize = 400;
% format
% frame, id, bb_left, bb_top, bb_w, bb_h, conf, x3d, y3d, z3d
% %%
% idx = 1;
% i = 1;
% framei = 1;
% Imean = double(imread(fullfile(imgdir, files(1).name)));
% 
% while framei<numel(files)
%     framei = framei + 1;
%     Imean = (Imean*(framei-1) + double(imread(fullfile(imgdir, files(framei).name))))/framei;
%     display(num2str(framei/numel(files)));
% end
% imshow(uint8(Imean));

%%
idx = 1;
i = 1;
framei = 0;
seq_det = [];
while framei<numel(files)
    framei = framei + 1;
    resi = tracks(tracks(:,1) == framei, :);
    resi = (resi(:,3:7));
    fname = fullfile(imgdir, files(framei).name);
    im0 = imread(fname);
    
    % filter by size
    resi = resi(resi(:,4)>minsize,:);
    resi = resi(resi(:,4)<maxsize,:);
    
    goodres = [];
    Idiff = backFiltM(im0, vidmode);
    for j = 1:size(resi,1)
        x = round(resi(j,1));
        y = round(resi(j,2));
        w = round(resi(j,3));
        h = round(resi(j,4));
        W = size(Idiff, 2);
        H = size(Idiff, 1);
        
        patchdiff = Idiff(max(y,1):min(y+h, H),max(x,1):min(x+w, W));
        normpatchdiff = sum(patchdiff(:)==1)/numel(patchdiff);
        if normpatchdiff > fthresh
            goodres(j) = true;
        else
            goodres(j) = false;
        end
    end
    resi = resi(logical(goodres)', :);

    
    bbs1 = bbNms(resi, 'thr', scorethresh, 'overlap', overlapthresh);
    
    seq_det = vertcat(seq_det, [(framei)*ones(size(bbs1,1), 1), -1*ones(size(bbs1,1), 1), bbs1, -1*ones(size(bbs1,1), 3)]);
    
    I = insertObjectAnnotation(im0,...
        'rectangle', ceil(bbs1(:,1:4)),...
        bbs1(:,5),...
        'TextBoxOpacity',0.7, 'LineWidth', 3);
    step(v,I)
    pause(1/300);
end

fid = fopen(detfile, 'w');
fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\r\n', seq_det');
fclose(fid);