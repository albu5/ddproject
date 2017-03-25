function viewDet(imgdir, trackfile)
% vidfile = 'C:\Users\Ashish\Desktop\DDP\agutils\data\road\road.avi';
% trackfile = 'C:\Users\Ashish\Desktop\DDP\agutils\data\road\track.txt';
tracks = csvread(trackfile);
c = round(256*rand(512,3));
cdata = [];
% format
% frame, id, bb_left, bb_top, bb_w, bb_h, conf, x3d, y3d, z3d
%%
files = dir([imgdir filesep '*.jp*']);
idx = 1;
i = 1;
framei = 0;
while framei<numel(files)
    frame = imread(fullfile(imgdir, files(framei+1).name));
    annDone = false;
    while ~annDone
        frame = insertObjectAnnotation(frame,'rectangle', ...
            round(tracks(i,3:6)),...
            num2str(tracks(i,7)),...
            'Color',c(rem(i,500)+1,:),...
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
    framei = framei + 1;
    imagesc(frame), axis image, axis off, title(num2str(idx));
    pause(1/30);
end