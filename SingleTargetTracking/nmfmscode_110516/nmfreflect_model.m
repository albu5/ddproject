% clear all;
% close all;
% clc;
function [Jill,J]=nmfreflect_model(I,rdim,img_size,t_size,s,max_iter)
% img_size = 256;
% t_size = 256;
% rdim = 2;
% s = 0.3;
% 
% max_iter = 50;
%str = 'H:\cvl\nmfsc\dat\Picture4.png';
%str = 'H:\cvl\nmfsc\data\pcover.jpg';
%str = 'N542.tif';
%str = 'syn.png';
% str='test2.jpg';
% REAL_ILLUM = load('REAL_ILLUM_900.txt');
% real_illum = REAL_ILLUM(33,:); real_illum=real_illum./sum(real_illum);
gamma = 1;

iptsetpref('ImshowBorder','tight');
%iptsetpref('ImviewInitialMagnification','fit');
%I = imread(str);
%I11=imread('0033.jpg');
%I(end*5/6:end,:,:)=[];
%I = double(imresize(I11,[img_size img_size]));
%I = repmat(mean(I,3),[1 1 3]);
%I = cat(3, I(:,:,1)*245,I(:,:,2)*224,I(:,:,3)*119)/255; 
% figure; 
% imshow(I./255);
% title('original image');

[n, m d] =size(I);
[t_sizex,t_sizey,z]=size(I);
n = floor(n/t_sizex)*t_sizex;
m = floor(m/t_sizey)*t_sizey;

gray_ill = squeeze(sum(sum(I)));
gray_ill = gray_ill./sum(gray_ill);


logI = abs(-log(double(imresize(I+1,[n m])/255).^(1/gamma)));

if rdim == 2
   % t_size = img_size;
    V = [im2col(logI(:,:,1),[t_sizex t_sizey]); im2col(logI(:,:,2),[t_sizex t_sizey]); im2col(logI(:,:,3),[t_sizex t_sizey])];
   % V = [V,V];
else
    V = [im2col(logI(:,:,1),[t_sizex t_sizey],'distinct'); im2col(logI(:,:,2),[t_sizex y],'distinct'); im2col(logI(:,:,3),[t_sizex t_sizey],'distinct')];
end

sW = ones(1,rdim)*s; sW(1) = 0.001;

[W, H] = nmfsc_RGB(V, rdim, sW, [], 'temp',0, gray_ill, max_iter);

expW = exp(-W(:,1)*mean(H(1,:)));
Jill = reshape(expW,[t_sizex t_sizey 3]);   
nmf_illum = squeeze(sum(sum(Jill))); nmf_illum = nmf_illum./sum(nmf_illum);
%J = J./max(J(:))*max(I(:))/255;
% figure(6); imshow(Jill);
% title('illuminated image');
% J = J./repmat(sum(J,3),[1 1 3]);
% J = cat(3, 1/3./J(:,:,1).*I(:,:,1), 1/3./J(:,:,2).*I(:,:,2), 1/3./J(:,:,3).*I(:,:,3));
% figure; imshow(J/255,[]);

% J = cat(3, 1/3./real_illum(1).*I(:,:,1), 1/3./real_illum(2).*I(:,:,2), 1/3./real_illum(3).*I(:,:,3));
% figure; imshow(J/255,[]);
%J = cat(3, 1/3/nmf_illum(1).*I(:,:,1), 1/3/nmf_illum(2).*I(:,:,2), 1/3/nmf_illum(3).*I(:,:,3));
%figure; imshow(imresize(J,1)/255,[])
for j = 2:size(W,2)    
    J = reshape(exp(-W(:,j)*mean(H(j,:))),[t_sizex t_sizey 3]);   
    J = (J-min(J(:)))./(max(J(:))-min(J(:)));
    J = J./repmat(sum(J,3),[1 1 3]);
    J = J./max(J(:));
%     figure(7); imshow(J);
end

