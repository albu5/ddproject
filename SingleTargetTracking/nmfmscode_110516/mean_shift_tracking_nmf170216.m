clc;
clear;
img_ind=input('enter starting image index = '); %enter starting image index
 %img_ind = 23;
%img_ind = 377;
%img_ind = 1900;
%img_ind=1255;
%img_ind=0130;
%img_ind=0154;
%img_ind=0010;
%img_ind=0176;
% img_ind=2400;
% img_ind=00595;
%img_ind=0264;
%img_ind=0809;
%img_ind=00814;
% img_ind=02410;
%img_ind=00760;
%img_ind=00573;
%img_ind=0444;
%read_dir=input('enter the directry where images are located in single quotes = ');%enter the directry where images are located
%read_dir = 'SECOND/SECOND ';
%read_dir='Video 1/Video 1 ';
%read_dir='WalkByShop1cor/WalkByShop1cor';
% read_dir='Walk/Walk';
 %read_dir='OneShopOneWait1cor/OneShopOneWait1cor';
% read_dir='OneShopOneWait2cor/OneShopOneWait2cor';
% read_dir='OneLeaveShop2cor/OneLeaveShop2cor';
 %read_dir='OneLeaveShopReenter2cor/OneLeaveShopReenter2cor';
 %read_dir='S2-T3-C/S2-T3-C';
 % read_dir='S3-T7-A/S3-T7-A';
% read_dir='S6-T3-H/S6-T3-H';
%read_dir='OneStopMoveEnter1cor/OneStopMoveEnter1cor';
% read_dir='ThreePastShop2cor/ThreePastShop2cor';
% read_dir='ThreePastShop1cor/ThreePastShop1cor';
%read_dir='TwoEnterShop2cor/TwoEnterShop2cor';
%read_dir='motinas_toni_change_ill\motinas_toni_change_ill ';
%read_dir='motinas_toni_change_ill\motinas_toni_change_ill ';

%read_dir='box\box';
%read_dir='fikerillu\fikerillu ';
%read_dir='darkillutwo/darkillutwo ';
%read_dir='frame_/frame_';
%read_dir='motinas_multi_face_frontal\motinas_multi_face_frontal ';
%read_dir='motinas_multi_face_turning\motinas_multi_face_turning ';
%read_dir='lemming/lemming';
%read_dir='MOV02369/MOV02369 ';
%read_dir='MOV02368/MOV02368 ';
%read_dir='shaking/frame_';
%read_dir='Walk/Walk';
%read_dir='Tiger1/img';

write_dir=input('enter the directry where results to be written in single quotes = ');%enter the directry where results to be written
% write_dir = 'result';
mkdir(write_dir);
%frame=sprintf('%s%d.jpg',read_dir,img_ind);%getting initial frame
%frame=sprintf('%s%d.png',read_dir,img_ind);
%frame=sprintf('%s%03d.jpg',read_dir,img_ind);
 % frame=sprintf('%s.%05d.jpeg',read_dir,img_ind);
% frame=sprintf('%s%06d.jpg',read_dir,img_ind);
 % frame=sprintf('%s %05d.jpg',read_dir,img_ind);
 %frame=sprintf('%s%07d.jpg',read_dir,img_ind);
% frame=sprintf('monti_illuvari/%d.jpeg' ,img_ind);
%frame=sprintf('%s%03d.jpg',read_dir,img_ind);%
%frame=sprintf('Tiger1/img/%04d.jpg',img_ind);
%frame=sprintf('%s%06d.jpg',read_dir,img_ind);
%frame=sprintf('Matrix/img/%04d.jpg',img_ind);
frame=sprintf('David/img/%04d.jpg',img_ind);
%frame=sprintf('N1_ARENA-01_02_ENV_RGB_3/%05d.jpg',img_ind);
%frame=sprintf('A1_ARENA-15_06_TRK_RGB_2/%05d.jpg',img_ind);
init_frame = imread(frame);
%init_frame=imresize(init_frame,[256,256]);
imshow(init_frame);
% I = im2double(init_frame);%covert rgb image to double format
% I =I*255; % t
center = ginput(1);%getting initial position of the center 
in_center = center;

I = im2double(init_frame);%covert rgb image to double format
I =I*255; % to get same values in indexed image

%bins=input('enter no of bins = ');% specify no of bins (better power of 2)
 bins = 16;

 h_x=input('enter width of the kernel = ');
 h_y=input('enter height of the kernel = ');
% h_x=15;
% h_y=15;

%%
%%%%illumination seperation
img_size = 256;  
t_size = 256;
rdim = 2;
s = 0.3;
max_iter=50;
[Jill,J]=nmfreflect_model(I,rdim,img_size,t_size,s,max_iter);
J1=J*255;
 q_u1=hist_model(J,bins,center,h_x,h_y);
 q_u=q_u1;
% %b=zeros(16,16);
% % for temp=1:1:16
% %     a=q_u(:,:,temp);
% %     [pc, zscores, pcvars] = princomp(a);
% %     b(16,temp)=pcvars;
% % end
% 
% 
% oldcenter=center;

temph(:,:,:,1)=q_u1;
bhat_coeff= bhattacharya_coeff(q_u1,q_u1,bins);
 frames=input('enter no frames to process = ');
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
    
    tic
    
   % read_frame=sprintf('%s%d.jpg',read_dir,img_ind);
%    read_frame=sprintf('%s%d.png',read_dir,img_ind);
 %read_frame=sprintf('%s%03d.jpg',read_dir,img_ind);
    %write_frame=sprintf('%s%d.png',write_dir,img_ind);
   %read_frame=sprintf('%s.%05d.jpeg',read_dir,img_ind);
 %read_frame=sprintf('%s%06d.jpg',read_dir,img_ind);
%     read_frame=sprintf('%s%d.jpg',read_dir,img_ind);
%read_frame=sprintf('%s %05d.jpg',read_dir,img_ind);
%read_frame=sprintf('%s%07d.jpg',read_dir,img_ind);
%read_frame=sprintf('%s%03d.jpg',read_dir,img_ind);%
read_frame=sprintf('David/img/%04d.jpg',img_ind);
%read_frame=sprintf('monti_illuvari/%d.jpeg' ,img_ind);
%read_frame=sprintf('%s%03d.jpg',read_dir,img_ind);
%read_frame=sprintf('Matrix/img/%04d.jpg',img_ind);
%read_frame=sprintf('Singer1/img/%04d.jpg',img_ind);
%read_frame=sprintf('N1_ARENA-01_02_ENV_RGB_3/%05d.jpg',img_ind);
%read_frame=sprintf('A1_ARENA-15_06_TRK_RGB_2/%05d.jpg',img_ind);
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
    [Jill,J]=nmfreflect_model(I,rdim,img_size,t_size,s,max_iter);
     J1=J*255;
    [oldcenter,bhat_coeff,p_u,w] = mean_shift(J1,center,q_u1,h_x,h_y,bins);
  % [oldcenter,bhat_coeff,p_u] = mean_shift_withoutiter(I,center,q_u,h_x,h_y,bins);
  
 q_u1=update_nmfmodel(temph,frameint,q_u1, bhat_coeff);

       frameint
%         end
 temph(:,:,:,frameint)=p_u;
 
 bhat_coeff_antena(frameint)=bhat_coeff;
 nmf_antena_cex(frameint)=oldcenter(1);
 nmf_antena_cey(frameint)=oldcenter(2);
    T(frameint)=toc;
    
%     T(i)=toc;
    
%    bhat(frameint)= bhat_coeff;
    center=oldcenter;
    
    I=draw_box(oldcenter(1),oldcenter(2),h_x+1,h_y+1,I);
    I=draw_box(oldcenter(1),oldcenter(2),h_x-1,h_y-1,I);
    I=draw_box(oldcenter(1),oldcenter(2),h_x,h_y,I);
    imwrite(I/255,write_frame);
   
end
 %%
% %movie2avi(M, 'three_bms_221212.avi');
%  %save bms_temp_cex12
%  % save bms_temp_cey12
% % % toc;
% % % t=toc;
% % time=sum(T);
% % figure
% % plot(T);
% % 
% % title('mean shift time plot');
% % grid on
% % xlabel('frame index');
% % ylabel('time taken');
%for k = 2:50
%I = imread(['' num2str(k) '.jpeg']);
%M(k) = im2frame(I);
%end
