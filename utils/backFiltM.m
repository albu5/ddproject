function bw4 = backFiltM(im0, vidmode, opts)

if ~exist('opts','var')
    opts.bwthresh = 10;
    opts.strel = 5;
    opts.minSize = 400;
end

imdiff = mean(abs(double(im0)-double(vidmode)),3);

bwthresh = opts.bwthresh;
sel = strel('disk', opts.strel);

bw1 = imdiff>bwthresh;
bw2 = imerode(bw1, sel);
bw3 = imdilate(bw2, sel);
bw4 = bwareaopen(bw3, opts.minSize);
