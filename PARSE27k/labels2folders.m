indir = 'PARSE224';
outdir = 'PARSE27K224';

%%
i = 1;
counters = ones(8,1);
labels = csvread(sprintf('%s/train/labels.txt',indir));
while true
    try
        img = sprintf('%s/train/%d.jpg',indir,i);
        outimg = sprintf('%s/train/%d/%d.jpg',outdir, labels(i),...
            counters(labels(i)));
        if (~exist(sprintf('%s/train/%d',outdir, labels(i)), 'dir'))
            mkdir(sprintf('%s/train/%d',outdir, labels(i)));
            display(sprintf('%s/train/%d',outdir, labels(i)))
        end
        copyfile(img, outimg);
        if rem(i,100)== 0, display(outimg); end
    catch
        break
    end
    counters(labels(i)) = counters(labels(i))+1;
    i = i+1;
end

%%
i = 1;
counters = ones(8,1);
labels = csvread(sprintf('%s/valid/labels.txt',indir));
while true
    try
        img = sprintf('%s/valid/%d.jpg',indir,i);
        outimg = sprintf('%s/valid/%d/%d.jpg',outdir, labels(i),...
            counters(labels(i)));
        if (~exist(sprintf('%s/valid/%d',outdir, labels(i)), 'dir'))
            mkdir(sprintf('%s/valid/%d',outdir, labels(i)));
            display(sprintf('%s/valid/%d',outdir, labels(i)))
        end
        copyfile(img, outimg);
        if rem(i,100)== 0, display(outimg); end
    catch
        break
    end
    counters(labels(i)) = counters(labels(i))+1;
    i = i+1;
end

%%
i = 1;
counters = ones(8,1);
labels = csvread(sprintf('%s/test/labels.txt',indir));
while true
    try
        img = sprintf('%s/test/%d.jpg',indir,i);
        outimg = sprintf('%s/test/%d/%d.jpg',outdir, labels(i),...
            counters(labels(i)));
        if (~exist(sprintf('%s/test/%d',outdir, labels(i)), 'dir'))
            mkdir(sprintf('%s/test/%d',outdir, labels(i)));
            display(sprintf('%s/test/%d',outdir, labels(i)))
        end
        copyfile(img, outimg);
        if rem(i,100)== 0, display(outimg); end
    catch
        break
    end
    counters(labels(i)) = counters(labels(i))+1;
    i = i+1;
end