
clc;
clear;
img_ind=input('enter starting image index = '); %enter starting image index

write_dir=input('enter the directry where results to be written in single quotes = ');%enter the directry where results to be written
% write_dir = 'result';
mkdir(write_dir);

frame=sprintf('blanket/%08d.jpg',img_ind);
init_frame = imread(frame);
imshow(init_frame);

center = ginput(1);%getting initial position of the center
h_x=input('enter width of the kernel = ');
h_y=input('enter height of the kernel = ');
region = [center(1)-h_x, center(2)-h_y, 2*h_x+1, 2*h_y+1];
frames=input('enter no frames to process = ');

I = im2double(init_frame);%covert rgb image to double format
I =I*255; % to get same values in indexed image

%init_frame=imresize(init_frame,[256,256]);
% I = im2double(init_frame);%covert rgb image to double format
% I =I*255; % t
%bins=input('enter no of bins = ');% specify no of bins (better power of 2)

% h_x=15;
% h_y=15;

%% illumination seperation

% some fixed params*****************
params.bins = 16;
params.rdim = 2;
params.s = 0.3;
params.max_iter=50;
% **********************************

[Jill,J] = nmfreflect_model2(I,params);
J1=J*255;

state.q_u1 = hist_model2(J,params.bins,region);
state.temph = state.q_u1;
state.bhat_coeff = bhattacharya_coeff(state.q_u1, state.temph, params.bins);

% % frames = 100;
% %frames = 159;
% %frames = 200;
% % tic
% trial1=zeros(frames,20);
%  trial2=zeros(frames,20);
%  k=1;
for frameint=2:1:frames
    
    img_ind = img_ind+1;
    % img_ind=img_ind-1;
    write_frame=sprintf('%s//%d.jpeg',write_dir,img_ind);
    
    %     tic
    
    
    read_frame=sprintf('blanket/%08d.jpg',img_ind);
    
    %    if bhat_coeff < 0.9
    %        q_u1= temph(:,:,:,frameint-1);
    %        bhat_coeff3= bhattacharya_coeff(p_u,q_u1,bins);
    %        display('update');
    %        frameint
    %         end
    %read_frame=sprintf('shaking/img/%04d.jpg',img_ind);
    
    init_frame = imread(read_frame);
    %  init_frame=imresize(init_frame,[256,256]);
    I = im2double(init_frame);
    I = I*255;
    [Jill,J] = nmfreflect_model2(I,params);
    J1=J*255;
    [region,state.bhat_coeff,state.temph,w] = mean_shift2(J1,state,region,params.bins);
    % [oldcenter,bhat_coeff,p_u] = mean_shift_withoutiter(I,center,q_u,h_x,h_y,bins);
    
    state.q_u1=update_nmfmodel2(state);
    
    h_x = (region(3)-1)/2;
    h_y = (region(4)-1)/2;
    center = [region(1)+h_x, region(2)+h_y];
    frameint
    I=draw_box(center(1),center(2),h_x+1,h_y+1,I);
    I=draw_box(center(1),center(2),h_x-1,h_y-1,I);
    I=draw_box(center(1),center(2),h_x,h_y,I);
    imwrite(I/255,write_frame);
    
end

