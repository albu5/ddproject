addpath(genpath('Helpers'));
thresh = -0.5;
% %%
% imgdir = '/home/viplab/Desktop/ped-det/data/temple/img';
% detfile = '/home/viplab/Desktop/ped-det/data/temple/det.txt';
% minsize = 2;
% maxsize = 6;
% deptest(imgdir, detfile, minsize, maxsize, thresh)
% 
% %%
% imgdir = '/home/viplab/Desktop/ped-det/data/temple2/img';
% detfile = '/home/viplab/Desktop/ped-det/data/temple2/det.txt';
% minsize = 2;
% maxsize = 6;
% deptest(imgdir, detfile, minsize, maxsize, thresh)
% 
% %%
% imgdir = '/home/viplab/Desktop/ped-det/data/road/img';
% detfile = '/home/viplab/Desktop/ped-det/data/road/det.txt';
% minsize = 2;
% maxsize = 6;
% deptest(imgdir, detfile, minsize, maxsize, thresh)
% 
% %%
% imgdir = '/home/viplab/Desktop/ped-det/data/road2/img';
% detfile = '/home/viplab/Desktop/ped-det/data/road2/det.txt';
% minsize = 2;
% maxsize = 6;
% deptest(imgdir, detfile, minsize, maxsize, thresh)
% 
% %%
% imgdir = '/home/viplab/Desktop/ped-det/data/road3/img';
% detfile = '/home/viplab/Desktop/ped-det/data/road3/det.txt';
% minsize = 2;
% maxsize = 6;
% deptest(imgdir, detfile, minsize, maxsize, thresh)
% 
% %%
% imgdir = '/home/viplab/Desktop/ped-det/data/court/img';
% detfile = '/home/viplab/Desktop/ped-det/data/court/det.txt';
% minsize = 2;
% maxsize = 6;
% deptest(imgdir, detfile, minsize, maxsize, thresh)
% % 
% % %%
% % imgdir = '/home/viplab/Desktop/ped-det/data/masjid/img';
% % detfile = '/home/viplab/Desktop/ped-det/data/masjid/det.txt';
% % minsize = 5;
% % maxsize = inf;
% % thresh = -0.5;
% % deptest(imgdir, detfile, minsize, maxsize, thresh)
% 
% %%
% imgdir = '/home/viplab/Desktop/ped-det/data/masjid2/img';
% detfile = '/home/viplab/Desktop/ped-det/data/masjid2/det.txt';
% minsize = 5;
% maxsize = inf;
% thresh = -0.5;
% deptest(imgdir, detfile, minsize, maxsize, thresh)

%%
imgdir = '../../data/3dmottc/img';
detfile = '../../data/3dmottc/detbad.txt';
minsize = 2;
maxsize = 6;
thresh = -0.5;
deptest(imgdir, detfile, minsize, maxsize, thresh, true, 'jp')

%%
imgdir = '/home/viplab/Desktop/ped-det/data/temple4/img';
detfile = '/home/viplab/Desktop/ped-det/data/temple4/det.txt';
minsize = 2;
maxsize = 6;
thresh = -0.5;
deptest(imgdir, detfile, minsize, maxsize, thresh)

%%
imgdir = '/home/viplab/Desktop/ped-det/data/temple5/img';
detfile = '/home/viplab/Desktop/ped-det/data/temple5/det.txt';
minsize = 2;
maxsize = 6;
thresh = -0.5;
deptest(imgdir, detfile, minsize, maxsize, thresh)

%%
imgdir = '/home/viplab/Desktop/ped-det/data/road4/img';
detfile = '/home/viplab/Desktop/ped-det/data/road4/det.txt';
minsize = 2;
maxsize = 6;
thresh = -0.5;
deptest(imgdir, detfile, minsize, maxsize, thresh)

%%

seqs = [];
success = [];
mins = [];
maxs = [];
ext = [];

seqs{1} = '3dmotpets1';
seqs{2} = '3dmotpets2';
seqs{3} = '3dmottc';
seqs{4} = '3dmottud';

success{1} = 0;
success{2} = 0;
success{3} = 0;
success{4} = 0;

mins{1} = 1;
mins{2} = 1;
mins{3} = 2;
mins{4} = 2;

maxs{1} = 6;
maxs{2} = 6;
maxs{3} = 10;
maxs{4} = 10;

ext{1} = 'jp';
ext{2} = 'jp';
ext{3} = 'jp';
ext{4} = 'jp';


parfor i = 1:4
    success{i} = deptest(['../../data/' seqs{i} '/img'], ...
        ['../../data/' seqs{i} '/detraw.txt'], ...
        mins{i}, maxs{i}, -0.5, false, ext{i});
end