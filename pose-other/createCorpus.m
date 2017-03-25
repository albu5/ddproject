load multiview_traindata.mat
load multiview_validdata.mat

%%
if ~exist('MULTIVIEW', 'dir')
   mkdir MULTIVIEW;
   mkdir MULTIVIEW/train;
   mkdir MULTIVIEW/valid;
end

%%
for i = 1:numel(trainy)
   im = reshape(trainx(i,:), 100, 40);
   imwrite(im, sprintf('MULTIVIEW/train/%d.png',i));
end
csvwrite('MULTIVIEW/train/labels.txt', trainy);

%%
for i = 1:numel(validy)
   im = reshape(validx(i,:), 100, 40);
   imwrite(im, sprintf('MULTIVIEW/valid/%d.png',i));
end
csvwrite('MULTIVIEW/valid/labels.txt', validy);
