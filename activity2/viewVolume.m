function h =  viewVolume(volume, fps)
if ~exist('fps','var')
    fps = 30;
end
h = figure;
for i = 1:size(volume,4)
   imagesc(volume(:,:,:,i)), axis image
   pause(1/fps)
end
end