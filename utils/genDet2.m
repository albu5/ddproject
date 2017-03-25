datadir = '../../data/eebuilding2';
toolboxpath = '../../external/toolbox-master/';
load(fullfile(toolboxpath, 'detector',  'models', 'AcfCaltech+Detector.mat'));
addpath(genpath(toolboxpath))
imgdir = [datadir filesep 'img'];
detfile = [datadir filesep 'det1.txt'];

%%
files = dir([imgdir filesep '*.jp*']);
framei = 0;
seq_det = [];
while framei<numel(files)
    framei = framei + 1;
    frame = imread(fullfile(imgdir, files(framei).name));
    bbox = acfDetect(frame, detector);
    for bb = bbox(:,1:4)'
        frame = insertObjectAnnotation(frame, 'Rectangle', bb', 'Person');
        imagesc(frame)
    end
    pause(0.001)
    seq_det = vertcat(seq_det, [framei*ones(size(bbox,1), 1) -1*ones(size(bbox,1), 1) bbox -1*ones(size(bbox,1), 3)]);
end
%%
fid = fopen(fullfile(datadir, 'det.txt'), 'w');
fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\r\n', seq_det');
fclose(fid);