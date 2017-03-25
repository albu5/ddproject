%function viewTrack(vidfile, trackfile)
imgdir = '../../data/eebuilding/img';
trackfile = '../../data/eebuilding/track.txt';
files = dir([imgdir filesep '*.jp*']);
tracks = csvread(trackfile);
c = round(256*rand(512,3));
cdata = [];
% format
% frame, id, bb_left, bb_top, bb_w, bb_h, conf, x3d, y3d, z3d
%%
idx = 1;
i = 1;
framei = 0;
while framei<numel(files)
        framei = framei + 1;
    frame = imread(fullfile(imgdir, files(framei).name));
    annDone = false;
    while ~annDone
        frame = insertObjectAnnotation(frame,'rectangle', ...
            round(tracks(i,3:6)),...
            num2str(tracks(i,2)),...
            'Color',c(tracks(i,2),:),...
            'LineWidth', 5, ...
            'TextBoxOpacity',0.8,...
            'FontSize',10);
        i = i+1;
        if i<=size(tracks,1)
            if tracks(i,1)>idx
                idx = idx + 1;
                annDone = true;
            end
        else break
        end
    end

    imagesc(frame), axis image, axis off, title(num2str(idx));
    pause(0.00001);
end
u.close()