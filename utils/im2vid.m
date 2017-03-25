function im2vid(impath, outpath, framerate, imft)

files = dir([impath filesep '*.' imft]);

v = VideoWriter(outpath);
v.FrameRate = framerate;
v.open();
for file = files'
    display(file.name)
    writeVideo(v,imread([impath filesep file.name]))
end
v.close();