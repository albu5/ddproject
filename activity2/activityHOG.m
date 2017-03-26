%%
datadir = '../../data/ActivityDataset/individuals_short';

classes = {'1','2'};

for i = classes
   classdir = fullfile(datadir, char(i));
   seqs = dir(classdir);
   seqs = seqs(3:end);
   for seq = seqs'
      fvec = [];
      seqdir = fullfile(classdir, seq.name);
      display(seqdir);
      imgs = dir([seqdir filesep '*.jp*']);
      for j = 1:numel(imgs)
%          temp = extractHOGFeatures(imread(fullfile(seqdir,sprintf('%d.jpg',j))));
        temp = imresize(imread(fullfile(seqdir,sprintf('%d.jpg',j))), [64 32]);
         fvec = vertcat(fvec,temp(:)');
      end
      csvname = fullfile(seqdir,'rgb.csv');
      csvwrite(csvname, fvec);
   end
end

%%
%%
datadir = '../../data/ActivityDataset/individuals30_split1';
splits = {'train', 'valid'};
classes = {'1','2'};

for split = splits
    for i = classes
        classdir = fullfile(datadir, char(split), char(i));
        seqs = dir(classdir);
        seqs = seqs(3:end);
        for seq = seqs'
            fvec = [];
            seqdir = fullfile(classdir, seq.name);
            display(seqdir);
            imgs = dir([seqdir filesep '*.jp*']);
            for j = 1:numel(imgs)
                         temp = extractHOGFeatures(imread(fullfile(seqdir,sprintf('%d.jpg',j))));
%                 temp = imresize(imread(fullfile(seqdir,sprintf('%d.jpg',j))), [64 32]);
                fvec = vertcat(fvec,double(temp(:)')/255);
            end
            csvname = fullfile(seqdir,'hog.csv');
            csvwrite(csvname, fvec);
        end
    end
end