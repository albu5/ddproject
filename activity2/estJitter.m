function [jy,jx] = estJitter(I1, I2, bbs1, bbs2)

bbs1 = round(bbs1);
bbs2 = round(bbs2);

bbs1(:,1) = max(1,bbs1(:,1));
bbs1(:,2) = max(1,bbs1(:,2));
bbs1(:,3) = max(size(I1,2),bbs1(:,3));
bbs1(:,4) = max(size(I1,1),bbs1(:,4));

bbs2(:,1) = max(1,bbs2(:,1));
bbs2(:,2) = max(1,bbs2(:,2));
bbs2(:,3) = max(size(I1,2),bbs2(:,3));
bbs2(:,4) = max(size(I1,1),bbs2(:,4));

for bb1 = bbs1
    I1(bb1(2):bb1(4),bb1(1):bb1(3))=0;
end
for bb2 = bbs2
    I2(bb2(2):bb2(4),bb2(1):bb2(3))=0;
end
imagesc([I1,I2]), axis image

jx = 0;
jy = 0;

N = round((size(I1,1)/40))*2;
M = round((size(I1,2)/40))*2;
bestscore = inf;
for i = 1:N
    i
for j = 1:M
    I3 = imtranslate(I1,[j-M/2,i-N/2]);
    score = I3(N:end-N, M:end-M) - I2(N:end-N, M:end-M);
    score = score(:)'*score(:);
    if score<bestscore
        bestscore = score;
        jx = j-M/2;
        jy = i-N/2;
    end 
end
end

%     
% 
% %%
% jx = 0;
% jy = 0;
% figure
% 
% F1 = fftshift(fft2((I1)));
% F2 = fftshift(fft2((I2)));
% 
% Q = F1./F2;
% A = angle(Q);
% imagesc(A)
% 
% qx = A(end/2, end/2-end/32:end/2+end/32);
% qy = A(end/2-end/32:end/2+end/32, end/2);
% 
% Xx = (1:numel(qx))-numel(qx)/2;
% Xy = (1:numel(qy))-numel(qy)/2;
% size(Xx)
% size(qx)
% bx = regress(qx(:), [ones(numel(qx),1) Xx(:)]);
% by = regress(qy(:), [ones(numel(qy),1) Xy(:)]);
% jx = bx(2)*size(I1,2)/2;
% jy = by(2)*size(I1,1)/2;
% 
% figure
% plot(A(end/2,end/2-end/32:end/2+end/32))
% figure
% plot(A(end/2-end/32:end/2+end/32, end/2))
% 
