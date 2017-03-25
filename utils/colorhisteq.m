function j = colorhisteq(i)

I = double(i);
G = mean(I,3)/255;
H = histeq(G);
R = I(:,:,1).*((H+eps)./(G+eps));
G = I(:,:,2).*((H+eps)./(G+eps));
B = I(:,:,3).*((H+eps)./(G+eps));

j = uint8(cat(3,R,G,B));
imshow(j)
