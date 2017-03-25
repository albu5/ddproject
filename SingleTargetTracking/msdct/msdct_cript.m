function msdct_script

clc
clear

img_ind=input('enter starting image index = '); %enter starting image index
read_dir=['bag' filesep];

write_dir = input('enter results directory in single quotes: ');
if strcmp(write_dir(end), '\') || strcmp(write_dir(end), '/')
    write_dir = [write_dir(1:end-1) filesep];
end
mkdir(write_dir);

%%
%0th Frame
frame=sprintf('%s%08d.jpg',read_dir,img_ind);
zeroeth_frame = imread(frame);0
I0 = 255*im2double(zeroeth_frame);%covert rgb image to double format

%1st Frame
frame=sprintf('%s%08d.jpg',read_dir,img_ind+1);
init_frame = imread(frame);
I = 255*im2double(init_frame);%covert rgb image to double format
imshow(init_frame);

% initialization region
center = ginput(1);%getting initial position of the center
in_center = center;
h_x=input('enter width of the kernel = ');
h_y=input('enter height of the kernel = ');
region = [center(1)-h_x, center(2)-h_y, 2*h_x+1, 2*h_y+1];
frames=input('enter no frames to process = ');

% some fixed params*********************
params.bins = 16;
% **************************************

% initialization************************
[state.dctwt_q,~,~]=dct_wt3(I,I0,center,h_x,h_y);
state.q_u=hist_model(I,params.bins,center,h_x,h_y);
% **************************************

for frameint=1:1:frames
    % updating indices
    img_ind=img_ind+1;
    img_ind_prev=img_ind-1;
    
    %write present and previous frame
    write_frame=sprintf('%s//%d.jpeg',write_dir,img_ind);
    tic;
    
    %read present and previous frame
    read_frame=sprintf('%s%08d.jpg',read_dir,img_ind);
    read_frame_prev=sprintf('%s%08d.jpg',read_dir,img_ind_prev);
    init_frame = imread(read_frame);
    prev_frame=imread(read_frame_prev);
    
    %Double format
    I = im2double(init_frame);
    I = I*255;
    I_prev = im2double(prev_frame);
    I_prev = I_prev*255;
    
    % matlab region to centre0
    h_x = (region(3)-1)/2;
    h_y = (region(4)-1)/2;
    center = [region(1)+h_x, region(2)+h_y];
    
    % tracker update
    
    [state.dctwt_p,~,~]=dct_wt3(I,I_prev,center,h_x,h_y);
    [center,~,state.p_u,~] = mean_shift_dct2(I,center,state.q_u,h_x,h_y,params.bins,state.dctwt_q,state.dctwt_p);
    rho = bhattacharya_coeff(state.q_u,state.p_u,params.bins);
    region = [center(1)-h_x, center(2)-h_y, 2*h_x+1, 2*h_y+1];

    I=draw_box_r(center(1),center(2),h_x+1,h_y+1,I);
    I=draw_box_r(center(1),center(2),h_x-1,h_y-1,I);
    I=draw_box_r(center(1),center(2),h_x,h_y,I);
    imwrite(I/255,write_frame);
end
