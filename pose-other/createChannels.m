validx = [];
validy = csvread('MULTIVIEW96x40/valid/labels.txt');

pChns.shrink = 1;
pChns.pColor.enabled = 0;

i = 0;
while true
   i = i+1;
   im = imread(sprintf('MULTIVIEW96x40/valid/%d.png', i));
   im3 = double(cat(3,im,im,im))/255;
   chns = chnsCompute(im3, pChns);
   chnsgmag = chns.data{1};
   chnsgdir = chns.data{2};
   catchns = cat(3,im,chnsgmag,chnsgdir(:,:,1),chnsgdir(:,:,2),chnsgdir(:,:,3),chnsgdir(:,:,4),chnsgdir(:,:,5),chnsgdir(:,:,6));
   catchns = reshape(catchns, 1, numel(catchns));
   validx = vertcat(validx, catchns);
   display(sprintf('%d',i));
end