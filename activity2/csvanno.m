%%
datadir = '../../data/ActivityDataset';
sequences = dir([datadir, filesep, 'seq*']);

%%
for idx = 1:44
    load(fullfile(datadir, 'annotations', sprintf('anno%2.2d.mat', idx)));
    counter = 1;
    annovec = [];
    idx
    for ped = anno.people
        t = ped.time';
        bbs = ped.sbbs';
        id = counter*ones(size(t));
        props = horzcat(t,id,bbs);
        annovec = vertcat(annovec, props);
        counter = counter + 1;
    end
    [~,order] = sort(annovec(:,1));
    annovec = annovec(order,:);
    csvwrite(fullfile(datadir, sprintf('seq%2.2d',idx), 'tracks.csv'), annovec);
end

%%


% %%
% showTracks(fullfile(datadir, [sequences(1).name, '/']), tracks)
% 
% %%
% showDets(fullfile(datadir, [sequences(1).name, '/']), dets)
