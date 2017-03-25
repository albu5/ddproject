

function res = demo1(TEST_IMAGE_0, TEST_IMAGE_1, minScale, maxScale)
tic
MODEL_SIZE     = [50 20.5];
IMAGE_SIZE     = size(rgb2gray(imread(TEST_IMAGE_0)));
NUM_OCTAVE     = 8;
MODEL_FILENAME = 'model.mat';
STRIDE         = 4;
BING_THRESH    = -0.032;
PED_THRESH     = -0.5;


%
% Load BING and pedestrain detector models
%
if ~exist(MODEL_FILENAME, 'file'), return;
else  model = load(MODEL_FILENAME); ped_model=model.ped_model; bing_model=model.bing_model; end

%
% Compute optical flow between two consecutive images
%
alpha = 0.01;
ratio = 0.8;
minWidth = 20;
nOuterFPIterations = 3;
nInnerFPIterations = 1;
nSORIterations = 20;
flowThreshold = 20;
para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

im0 = imread(TEST_IMAGE_0); im1 = imread(TEST_IMAGE_1);
if ~isequal([size(im0,1),size(im0,2)], IMAGE_SIZE),
    im0 = imresize(im0, IMAGE_SIZE);
    im1 = imresize(im1, IMAGE_SIZE);
end
[vx,vy,~] = Coarse2FineTwoFrames(im0, im1, para);
t=toc; %fprintf('Extracting optical flow took: %.2f secs\n', t);
flow = cat(3,vx,vy);
flow=min(flowThreshold, flow); flow=max(-flowThreshold,flow);
flow=single(flow./flowThreshold); % flow image (single)
I   = im2single(im0);  % test image (single)
Iuint8 = im0;          % test image (uint8)


% Pre-compute BING masks at different scales
bing = evalBINGMex(Iuint8,bing_model); bing = bing(24:-1:1);
t=toc; %fprintf('Applying BING took: %.2f secs\n', t);


% get scales at which to compute features and list of real/approx scales
[scales,scaleshw]=getScales(NUM_OCTAVE,0,MODEL_SIZE,4,IMAGE_SIZE);
nScales=length(scales);

validScale = (scales>(1/maxScale)) & (scales<(1/minScale));

bbs = cell(nScales,1);
for i=1:nScales
    if (validScale(i))
        if i>length(bing), continue; end
        sc=scales(i); sc1=round(IMAGE_SIZE*sc/4); sc2=sc1*4;
        
        if size(I,1) ~= sc2(1) && size(I,2) ~= sc2(2)
            I1=imResampleMex(I,sc2(1),sc2(2),1); flow1=imResampleMex(flow,sc2(1),sc2(2),1);
        else
            I1=I; flow1=flow;
        end
        
        mask = zeros(sc1,'single');
        [h1,w1]=size(bing{i});
        if h1>sc1(1),h1=sc1(1); end, if w1>sc1(2),w1=sc1(2); end
        mask(1:h1,1:w1)=bing{i}(1:h1,1:w1);
        
        bb=detectPedMex(I1,flow1,mask,ped_model,STRIDE,PED_THRESH,BING_THRESH);
        
        if ~isempty(bb),
            bb(:,1) = (bb(:,1))./scaleshw(i,2);
            bb(:,2) = (bb(:,2))./scaleshw(i,1);
            bb(:,3:4) = bb(:,3:4)./sc;
            bbs{i}=bb;
        end
    end
end
t=toc;
fprintf('Applying pedestrian detector took: %.2f secs\n', t);

bbs=cat(1,bbs{:});
bbs=bbNms(bbs,'type','maxg','overlap',0.65,'ovrDnm','min');
res = bbs;


