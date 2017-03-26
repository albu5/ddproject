function [row, col] = estJitter2(I1, I2, bbs1, bbs2)
%I,I2 are reference and target images
%[row col] are row, column shifts
%SCd 4/2010
%
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

%Fourier transform both images
fi = fft2(double(I1));
fr = fft2(double(I2));

%Perform phase correlation (amplitude is normalized)
fc = fi .* conj(fr);
fcn = fc ./abs(fc);

%Inverse fourier of peak correlation matrix and max location
peak_correlation_matrix = abs(ifft2(fcn));
[peak, idx] = max(peak_correlation_matrix(:));

%Calculate actual translation
[row, col] = ind2sub(size(peak_correlation_matrix),idx);
if row < size(peak_correlation_matrix,1)/2
    row = -(row - 1);
else
    row = size(peak_correlation_matrix,1) - (row - 1);
end;
if col < size(peak_correlation_matrix,2)/2
    col = -(col - 1);
else
    col = size(peak_correlation_matrix,2) - (col - 1);
end
end