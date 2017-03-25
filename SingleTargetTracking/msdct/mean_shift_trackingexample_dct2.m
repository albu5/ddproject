clc;
clear all;
close all;
img_ind=input('enter starting image index = '); %enter starting image index
%read_dir='frame_/frame_;
% read_dir='david/david';
% read_dir='ironman/ironman';
% read_dir='shadow/shadow';
% read_dir='girl/girl';
% read_dir='tiger/tiger';
% read_dir='trellis/trellis';
% read_dir='bike/bike';
read_dir='bag\';
% read_dir='OneLeaveShopReenter1cor/OneLeaveShopReenter1cor';
write_dir=input('enter the directry where results to be written in single quotes = ');%enter the directry where results to be written
mkdir(write_dir);
%0th Frame
frame=sprintf('%s%08d.jpg',read_dir,img_ind-1);
zeroeth_frame = imread(frame);
% imshow(zeroeth_frame);
% center0 = ginput(1);%getting initial position of the center 
I0 = im2double(zeroeth_frame);%covert rgb image to double format
I0 =I0*255; % to get same values in indexed image
%1st Frame
frame=sprintf('%s%08d.jpg',read_dir,img_ind);
init_frame = imread(frame);
imshow(init_frame);
center = ginput(1);%getting initial position of the center 
in_center = center;
I = im2double(init_frame);%covert rgb image to double format
I =I*255; % to get same values in indexed image

 bins = 16;
 h_x1=input('enter width of the kernel = ');
 h_y1=input('enter height of the kernel = ');

[dctwt_q,pres_frbg,prev_frbg]=dct_wt3(I,I0,center,h_x1,h_y1);
q_u=hist_model(I,bins,center,h_x1,h_y1);


oldcenter=center;
frames=input('enter no frames to process = ');

trial1=zeros(frames,20);
trial2=zeros(frames,20);

for frameint=1:1:frames
    img_ind=img_ind+1;
    img_ind_prev=img_ind-1;
    %write present and previous frame
    write_frame=sprintf('%s//%d.jpeg',write_dir,img_ind);
    write_frame_prev=sprintf('%s//%d.jpeg',write_dir,img_ind_prev);
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
    [dctwt_p,pres_frbg,prev_frbg]=dct_wt3(I,I_prev,center,h_x1,h_y1);
    %Mean Shift
    [oldcenter,bhat_coeff,p_u,w] = mean_shift_dct2(I,center,q_u,h_x1,h_y1,bins,dctwt_q,dctwt_p);
    rho = bhattacharya_coeff(q_u,p_u,bins);
    bhat_coeff_1(frameint)=rho;
    T(frameint)=toc;
    center=oldcenter;
    
    I=draw_box_r(oldcenter(1),oldcenter(2),h_x1+1,h_y1+1,I);
    I=draw_box_r(oldcenter(1),oldcenter(2),h_x1-1,h_y1-1,I);
    I=draw_box_r(oldcenter(1),oldcenter(2),h_x1,h_y1,I);
    imwrite(I/255,write_frame);
end
