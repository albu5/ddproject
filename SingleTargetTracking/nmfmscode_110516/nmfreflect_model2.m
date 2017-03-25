function [Jill,J]=nmfreflect_model2(I,params)
rdim = params.rdim;
s = params.s;
max_iter= params.max_iter;
gamma = 1;

iptsetpref('ImshowBorder','tight');

[n, m, ~] =size(I);
[t_sizex,t_sizey,~]=size(I);
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
for j = 2:size(W,2)
    J = reshape(exp(-W(:,j)*mean(H(j,:))),[t_sizex t_sizey 3]);
    J = (J-min(J(:)))./(max(J(:))-min(J(:)));
    J = J./repmat(sum(J,3),[1 1 3]);
    J = J./max(J(:));
    %     figure(7); imshow(J);
end
end

