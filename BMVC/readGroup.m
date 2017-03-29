data_dir = '/home/ashish/Desktop/ped-det/data/ActivityDataset';
anno_dir = fullfile(data_dir, 'my_anno');
new_anno_dir = fullfile(data_dir, 'anno_ag');
% exist(anno_dir, 'dir')
pre_anno = 'data_%2.2d';
pre_seq = 'seq%2.2d';

%%
trainX = [];
trainY = [];
testX = [];
testY = [];
Table = [];
mkdir(new_anno_dir)

%%
for idx = 1:44
    anno = (load(fullfile(new_anno_dir, sprintf(pre_anno, idx))));
    display((fullfile(new_anno_dir, sprintf(pre_anno, idx))));
    anno = anno.anno;
    n_ped = numel(anno.people);
    for i = 1:n_ped
        for j = 1:n_ped
            if i ~= j
                for t = 1:10:numel(anno.people(1).action)
                    bbi = anno.people(i).bbs(t,:);
                    bbj = anno.people(j).bbs(t,:);
                    if sum(bbi)>0 && sum(bbj)>0
                        pedi = anno.people(i);
                        pedj = anno.people(j);
                        fi = [pedi.pose(t),...
                            pedi.action(t),...
                            pedi.group_activity(t),...
                            pedi.bbs(t,:),...
                            pedi.group_label(t)];
                        fj = [pedj.pose(t),...
                            pedj.action(t),...
                            pedj.group_activity(t),...
                            pedj.bbs(t,:),...
                            pedj.group_label(t)];
                        if numel(Table) == 0
                            Table = [fi, fj];
                        else
                            Table = vertcat(Table, [fi, fj]);
                        end
                    end
                end
            end
        end
    end
end