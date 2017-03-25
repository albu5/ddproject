function myDetGen(vidpath, vidfile)
% vidpath = 'C:\Users\Ashish\Desktop\DDP\datasets\pune-ashish\temple30.mp4';
v = VideoReader(fullfile(vidpath, vidfile));
%%
seq_det = [];
idx =  1;
while hasFrame(v)
    frame = (readFrame(v));
    [bbox, score] = detectPeopleACF(frame,  'Model', 'caltech-50x21', ...
        'MinSize', 3*[50 21], 'MaxSize', 5*[50 21], ...
        'NumScaleLevels', 8, ...
        'Threshold', -2);
    
    for bb = bbox(:,1:4)'
        frame = insertObjectAnnotation(frame, 'Rectangle', bb', 'Person');
    end
    pause(0.001)
    imagesc(frame), axis image, title(num2str(idx))
    seq_det = vertcat(seq_det, [idx*ones(size(bbox,1), 1) -1*ones(size(bbox,1), 1) bbox score -1*ones(size(bbox,1), 3)]);
    idx = idx + 1;
end
%%
fid = fopen(fullfile(vidpath, 'det.txt'), 'w');
fprintf(fid, '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\r\n', seq_det');
fclose(fid);