%function viewTrack(vidfile, trackfile)
vidfile = 'C:\Users\Ashish\Desktop\DDP\agutils\data\road\road.avi';
trackfile = 'C:\Users\Ashish\Desktop\DDP\agutils\data\road\track.txt';
outfile = 'C:\Users\Ashish\Desktop\DDP\agutils\data\road\road_dct.avi';
v = VideoReader(vidfile);
u = VideoWriter(outfile);
u.FrameRate = 30;
u.open();
tracks = csvread(trackfile);
c = round(256*rand(512,3));
cdata = [];
% format
% frame, id, bb_left, bb_top, bb_w, bb_h, conf, x3d, y3d, z3d
%%
idx = 1;
i = 1;
framei = 0;
while hasFrame(v)
    frame = readFrame(v);
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
    framei = framei + 1;
    if(framei>150)&&(framei<400)
        writeVideo(u,frame);
    end
    imagesc(frame), axis image, axis off, title(num2str(idx));
    pause(0.00001);
end
u.close()