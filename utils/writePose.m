function [Iw,theta] = writePose(currtrack, I, Bx, By, outdir, sequence)

theta = (atan2(By(2),Bx(2)));
idx = currtrack(1);
id = currtrack(2);
x = max(1,currtrack(3));
y = max(1,currtrack(4));
w = min(size(I,2)-x,currtrack(5));
h = min(size(I,1)-y,currtrack(6));
Iw = I(round(y:y+h),round(x:x+w),:);


if theta>=-pi/8 && theta<pi/8
    imwrite(Iw, [outdir '0\' sprintf('%s-%d-%d.png', sequence, idx, id)]);
    
elseif theta>=pi/8 && theta<3*pi/8
    imwrite(Iw, [outdir '45\' sprintf('%s-%d-%d.png', sequence, idx, id)]);
    
elseif theta>=3*pi/8 && theta<5*pi/8
    imwrite(Iw, [outdir '90\' sprintf('%s-%d-%d.png', sequence, idx, id)]);
    
elseif theta>=5*pi/8 && theta<7*pi/8
    imwrite(Iw, [outdir '135\' sprintf('%s-%d-%d.png', sequence, idx, id)]);
    
elseif theta>=-3*pi/8 && theta<-pi/8
    imwrite(Iw, [outdir '315\' sprintf('%s-%d-%d.png', sequence, idx, id)]);
    
elseif theta>=-5*pi/8 && theta<-3*pi/8
    imwrite(Iw, [outdir '270\' sprintf('%s-%d-%d.png', sequence, idx, id)]);
    
elseif theta>=-7*pi/8 && theta<-5*pi/8
    imwrite(Iw, [outdir '225\' sprintf('%s-%d-%d.png', sequence, idx, id)]);
    
else
    imwrite(Iw, [outdir '180\' sprintf('%s-%d-%d.png', sequence, idx, id)]);
    
end