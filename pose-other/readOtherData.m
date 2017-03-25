datadir = '../../../data/3dmotpets2/';
outdir = [datadir 'persons'];

if ~exist(outdir, 'dir')
    mkdir(outdir);
end

model = [96,40];
tracks = csvread(fullfile(datadir, 'gt.txt'));
N = size(tracks,1);

counter = 0;
for i = 1:N
    counter = counter + 1;
    image = double(imread(fullfile(datadir, 'img', sprintf('%06d.jpg', tracks(i,1)))));
    if (size(image,3) == 3)
        image = mean(image, 3);
    end
    
    annorect.x1 = tracks(i,3);
    annorect.y1 = tracks(i,4);
    annorect.x2 = annorect.x1 + tracks(i,5);
    annorect.y2 = annorect.y1 + tracks(i,6);
    
    bbh = annorect.y2-annorect.y1;
    bbw = annorect.x2-annorect.x1;
    scale = max(bbh/model(1), bbw/model(2));
    tempim = imresize(image,1/scale);
    midx = floor((annorect.x2+annorect.x1)/(2*scale));
    midy = floor((annorect.y2+annorect.y1)/(2*scale));
    x = midx-model(2)/2+1:midx+model(2)/2;
    y = midy-model(1)/2+1:midy+model(1)/2;
    xprepad = 0;
    yprepad = 0;
    xpostpad = 0;
    ypostpad = 0;
    
    if midx-model(2)/2+1<1
        x = x(x>=1);
        xprepad = -midx+model(2)/2;
    end
    
    if midx+model(2)/2 > size(tempim,2)
        x = x(x<=size(tempim,2));
        xpostpad = midx+model(2)/2 - size(tempim,2);
    end
    
    if midy-model(1)/2+1<1
        y = y(y>=1);
        yprepad = -midy + model(1)/2;
    end
    if midy+model(1)/2 > size(tempim,1)
        y = y(y<=size(tempim,1));
        ypostpad = midy+model(1)/2 - size(tempim,1);
    end
    
    croppedim = tempim(y,x);
    if xprepad>0
        croppedim = padarray(croppedim, [0 xprepad 0], 'replicate', 'pre');
    end
    if xpostpad>0
        croppedim = padarray(croppedim, [0 xpostpad 0], 'replicate', 'post');
    end
    if yprepad>0
        croppedim = padarray(croppedim, [yprepad 0 0], 'replicate', 'pre');
    end
    if ypostpad>0
        croppedim = padarray(croppedim, [ypostpad 0 0], 'replicate', 'post');
    end
    imwrite(croppedim/255, fullfile(outdir, sprintf('%d.png',counter)));
end