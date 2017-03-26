datadir = '/home/ashish/Desktop/ped-det/data/ActivityDataset';
mkdir(fullfile(datadir, 'orientations'))
for i=1:44
    fname = fullfile(datadir, sprintf('seq%2.2d', i), 'orientations.csv');
    movefile(fname, fullfile(datadir, sprintf('orientations/seq%2.2d.csv',i)));
end