datadir = '/home/ashish/Desktop/ped-det/data/ActivityDataset/individuals30_split1/';

fvid1 = fullfile(datadir, 'train', '1', '890');
fvid2 = fullfile(datadir, 'train', '2', '155');
%%
vid1 = readVolume(fvid1);
viewVolume(vid1);

%%
vid2 = readVolume(fvid2);
viewVolume(vid2);

%%
close all
opticFlow = opticalFlowFarneback();
% i1 = mean(vid1(:,:,:,1),3);
for i = 1:size(vid1,4)
    i1 = mean(vid1(:,:,:,i),3);
    flow = estimateFlow(opticFlow,i1);
    dispI = ones([size(flow.Magnitude) 3]);
    dispI(:,:,1) = (flow.Orientation-min(flow.Orientation(:)))/(max(flow.Orientation(:))-min(flow.Orientation(:)));
    dispI(:,:,3) = (flow.Magnitude-min(flow.Magnitude(:)))/(max(flow.Magnitude(:))-min(flow.Magnitude(:)));
    imshow(imresize(hsv2rgb(dispI),4))
%     hold on
%     plot(flow,'DecimationFactor',[5 5],'ScaleFactor',10)
%     hold off
    pause(0.5)
end