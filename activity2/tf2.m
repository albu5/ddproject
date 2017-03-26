datadir = '/home/ashish/Desktop/ped-det/data/ActivityDataset/';
seqn = 30;
Nreset = 25;
opticFlow = opticalFlowHS();
addpath HOOF
close all
%%
for fnum = 1:100
    seqdir = fullfile(datadir,sprintf('seq%2.2d',seqn));
    imgs = dir([seqdir filesep '*.jpg']);
    
    if (rem(fnum-1,Nreset)==0)
        I1 = rgb2gray(double(imread(fullfile(seqdir, imgs(fnum).name)))/255);
        jxp = 0;
        jyp = 0;
    end
    
    I11 = rgb2gray(double(imread(fullfile(seqdir, imgs(fnum+1).name)))/255);
    I2 = rgb2gray(double(imread(fullfile(seqdir, imgs(fnum+1).name)))/255);
    
    load(fullfile(datadir, 'annotations', sprintf('anno%2.2d.mat', seqn)));
    people = anno.people;
    
    bbs1 = [];
    bbs2 = [];
    for ped = people
        it1 = find(ped.time == fnum);
        bb1 = ped.sbbs(:,it1);
        bbs1 = horzcat(bbs1,bb1);
        
        it2 = find(ped.time == fnum+1);
        bb2 = ped.sbbs(:,it2);
        bbs2 = horzcat(bbs2,bb2);
    end
    
    bbs1(3,:) = bbs1(3,:) + bbs1(1,:);
    bbs1(4,:) = bbs1(4,:) + bbs1(2,:);
    
    bbs2(3,:) = bbs2(3,:) + bbs2(1,:);
    bbs2(4,:) = bbs2(4,:) + bbs2(2,:);
    [jx,jy] = estJitter3(I1,I2,bbs1,bbs2);
    I22 = imtranslate(I2,[jx,jy]);
    
    ped = people(3);
    it1 = find(ped.time == fnum);
    bb1 = ped.sbbs(:,it1);
    bb1(1) = bb1(1) + jxp;
    bb1(2) = bb1(2) + jyp;
    
    it2 = find(ped.time == fnum+1);
    bb2 = ped.sbbs(:,it2);
    bb2(1) = bb2(1) + jx;
    bb2(2) = bb2(2) + jy;
    
    
    flow = estimateFlow(opticFlow,I22);
    dispI = ones([size(flow.Magnitude) 3]);
    dispI(:,:,1) = (flow.Orientation-min(flow.Orientation(:)))/(max(flow.Orientation(:))-min(flow.Orientation(:)));
    dispI(:,:,3) = (flow.Magnitude-min(flow.Magnitude(:)))/(max(flow.Magnitude(:))-min(flow.Magnitude(:)));
    
    Vx_ped = imcrop(flow.Vx,bb2');
    Vy_ped = imcrop(flow.Vy,bb2');
    Vx_leg = Vx_ped(end/2:end,:);
    Vy_leg = Vy_ped(end/2:end,:);
    
    myhist = gradientHistogram(Vx_leg, Vy_leg, 8);
    bar(myhist), ylim([0 100])
%     imshow(imresize([ hsv2rgb(imcrop(dispI,bb2')) imcrop(cat(3,I22,I22,I22),bb2')],4))

    % imshow([I2 I22])
    pause(0.1)
    
    jxp = jx;
    jyp = jy;
end