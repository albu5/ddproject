function vid2im(vidpath, vidfile, imft)
v = VideoReader(fullfile(vidpath, vidfile));

idx = 1;
mkdir([vidpath filesep 'img'])
while hasFrame(v)
    frame = readFrame(v);
    imwrite(frame, fullfile(vidpath, 'img', [sprintf('%08d', idx) '.' imft]));
    idx = idx + 1;
end
