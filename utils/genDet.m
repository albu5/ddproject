function genDet(vidpath, outpath, toolboxpath)
% vidpath = 'C:\Users\Ashish\Desktop\datasets\PETS09-S2L1\PETS09-S2L1.avi';
try
    addpath(genpath(toolboxpath))
catch err
    display(err.message)
    toolboxpath = '..\..\external\toolbox-master';
    addpath(genpath(toolboxpath))
end

load(fullfile(toolboxpath, 'detector',  'models', 'AcfCaltech+Detector.mat'));
% load([toolboxpath 'detector\models\AcfInriaDetector.mat']);
% load([toolboxpath 'detector\models\LdcfCaltechDetector.mat']);
% load([toolboxpath 'detector\models\LdcfInriaDetector.mat'');
detector.opts.pNms.thr = -inf;
detector.opts.pNms.overlap = 0.99;
detector.opts.stride = 1;
v = VideoReader(vidpath);
%%
seq_det = [];
idx =  1;
while hasFrame(v)
    frame = readFrame(v);
    bbox = acfDetect(frame, detector);
    for bb = bbox(:,1:4)'
        frame = insertObjectAnnotation(frame, 'Rectangle', bb', 'Person');
    end
    
    imagesc(frame)
    pause(0.001)
    seq_det = vertcat(seq_det, [idx*ones(size(bbox,1), 1) -1*ones(size(bbox,1), 1) bbox -1*ones(size(bbox,1), 3)]);
    idx = idx + 1;
end
%%
fid = fopen(fullfile(outpath, 'det.txt'), 'w');
fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\r\n', seq_det');
fclose(fid);