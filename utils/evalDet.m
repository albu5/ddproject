function [Q,MR] = evalDet(resfile, gtfile)
vres = csvread(resfile);
vgt = csvread(gtfile);
Q = zeros(1,max(vres(:,1)));
MR = zeros(1,max(vres(:,1)));

parfor i = 1:max(vres(:,1))
    res = vres(vres(:,1)==i+1,:);
    gt = vgt(vgt(:,1)==i,:);
    [a, mr] = (evalDetSingle(res,gt));
    Q(i) = mean(a);
    MR(i) = mr;
    if (rem(i,10) == 0)
        display(sprintf('%d frames completed of %d frames', i, max(vres(:,1))));
    end
end
end

function [q,missrate] = evalDetSingle(res,gt)
q = zeros(1,size(res,1));
missedgt = ones(size(gt,1),1);
for i = 1:size(res,1)
    res1 = res(i,:);
    tempq = zeros(1,size(gt,1));
    for j = 1:size(gt,1)
        gt1 = gt(j,:);
        tempq(j) = bboverlap(res1, gt1);
    end
    [q(i), gtidx] = max(tempq);
    missedgt(gtidx) = 0;
end
missrate = sum(missedgt);
end

function ovlap = bboverlap(res1, gt1)
im = zeros(ceil(max(res1(4)+res1(6), gt1(4)+gt1(6))), ceil(max(res1(3)+res1(5), gt1(3)+gt1(5))));
imr = im;
img = im;
imr(max(ceil(res1(4)),1):floor(res1(4)+res1(6)), max(1,ceil(res1(3))):floor(res1(3)+res1(5))) = 1;
img(max(1,ceil(gt1(4))):floor(gt1(4)+gt1(6)), max(1,ceil(gt1(3))):floor(gt1(3)+gt1(5))) = 1;
% imshow([imr|img])
% pause
ovlap = sum(imr(:)&img(:))/sum(imr(:)|img(:));
end