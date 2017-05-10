data_dir = './ActivityDataset';
anno_dir = fullfile(data_dir, 'Final_annotations');
% exist(anno_dir, 'dir')
pre_anno = 'data_%2.2d';
pre_seq = 'seq%2.2d';
out_dir = fullfile(data_dir, 'group_data');

%%
mkdir(out_dir);
trainX = [];
trainY = [];
testX = [];
testY = [];
colorstrs = ['r', 'g', 'b', 'm', 'k', 'c'];
testseq = [1     4     5     6     8     2     7    28    35    11    10    26];
%%
for seqi = 1:44
    if any(testseq == seqi)
        continue
    end
    seqdir = fullfile(data_dir, sprintf(pre_anno, seqi));
    annofile = fullfile(anno_dir, sprintf(pre_anno, seqi));
    anno = load(annofile);
    anno = anno.anno_data;
    t = 31;
    group_lab = anno.groups.grp_label;
    group_act = anno.groups.grp_act;
    while t < anno.nframe
        
        curr_groups = group_lab(t,:);
        curr_group_activites = group_act(t, :);
        n_groups = max(curr_groups);
        for i = 1:n_groups
            good_group = true;
            group_activity_i = curr_group_activites(i);
            people_i = anno.people(curr_groups == i);
            feat_vec = [];
            for person = people_i
                bbs = person.bbs(t-29:t, :);
                bbs(:, 1) = bbs(:, 1) + bbs(:, 3)/2;
                bbs(:, 2) = bbs(:, 2) + bbs(:, 4);
                pose = person.pose(t-29:t);
                action = person.action(t-29:t);
                validt = sum(bbs, 2) > 0;
                if sum(validt) > 10
                    t_vec = t-29:t;
                    t_vec = t_vec(validt);
                    pose = oneHot(mode(pose(validt)), 8);
                    action = oneHot(mode(action(validt)), 2);
                    bbs = bbs(validt, :);
                    bias = ones(size(bbs,1),1);
                    predictor = [bias, t_vec(:)];
                    [B2,~,R2] = regress(bbs(:,1),predictor);
                    [B3,~,R3] = regress(bbs(:,2),predictor);
                    [B4,~,R4] = regress(bbs(:,3),predictor);
                    [B5,~,R5] = regress(bbs(:,4),predictor);
                    meanbbs = bbs(end, :);
                    fvec = [B2(2), B3(2), B4(2), B5(2), pose, action, meanbbs];
                    if numel(feat_vec) == 0
                        feat_vec = fvec;
                    else
                        feat_vec = vertcat(feat_vec, fvec);
                    end
                end
            end
%             size(feat_vec, 1)
            if size(feat_vec, 1) < 10 && size(feat_vec, 1) > 0
                feat_vec = group_activity_feature(feat_vec);
%                 display('padding')
            elseif size(feat_vec, 1) > 10
                feat_vec = group_activity_feature(feat_vec);
%                 display('trimming')
            else
%                 display('skipping')
                good_group = false;
            end
            if good_group
                if numel(trainX) == 0
                    trainX = feat_vec(:)';
                    trainY(end+1) = group_activity_i;
                else
                    trainX(end+1, :) = feat_vec(:)';
                    trainY(end+1) = group_activity_i;
                end
            end
        end
        t = t + 10;
    end
end
trainY = trainY';

%%
for seqi = 1:44
    if ~any(testseq == seqi)
        continue
    end
    seqdir = fullfile(data_dir, sprintf(pre_anno, seqi));
    annofile = fullfile(anno_dir, sprintf(pre_anno, seqi));
    anno = load(annofile);
    anno = anno.anno_data;
    t = 31;
    group_lab = anno.groups.grp_label;
    group_act = anno.groups.grp_act;
    while t < anno.nframe
        
        curr_groups = group_lab(t,:);
        curr_group_activites = group_act(t, :);
        n_groups = max(curr_groups);
        for i = 1:n_groups
            good_group = true;
            group_activity_i = curr_group_activites(i);
            people_i = anno.people(curr_groups == i);
            feat_vec = [];
            for person = people_i
                bbs = person.bbs(t-29:t, :);
                bbs(:, 1) = bbs(:, 1) + bbs(:, 3)/2;
                bbs(:, 2) = bbs(:, 2) + bbs(:, 4);
                pose = person.pose(t-29:t);
                action = person.action(t-29:t);
                validt = sum(bbs, 2) > 0;
                if sum(validt) > 10
                    t_vec = t-29:t;
                    t_vec = t_vec(validt);
                    pose = oneHot(mode(pose(validt)), 8);
                    action = oneHot(mode(action(validt)), 2);
                    bbs = bbs(validt, :);
                    bias = ones(size(bbs,1),1);
                    predictor = [bias, t_vec(:)];
                    [B2,~,R2] = regress(bbs(:,1),predictor);
                    [B3,~,R3] = regress(bbs(:,2),predictor);
                    [B4,~,R4] = regress(bbs(:,3),predictor);
                    [B5,~,R5] = regress(bbs(:,4),predictor);
                    meanbbs = bbs(end, :);
                    fvec = [B2(2), B3(2), B4(2), B5(2), pose, action, meanbbs];
                    if numel(feat_vec) == 0
                        feat_vec = fvec;
                    else
                        feat_vec = vertcat(feat_vec, fvec);
                    end
                end
            end
%             size(feat_vec, 1)
            if size(feat_vec, 1) < 10 && size(feat_vec, 1) > 0
                feat_vec = group_activity_feature(feat_vec);
%                 display('padding')
            elseif size(feat_vec, 1) > 10
                feat_vec = group_activity_feature(feat_vec);
%                 display('trimming')
            else
%                 display('skipping')
                good_group = false;
            end
            if good_group
                if numel(testX) == 0
                    testX = feat_vec(:)';
                    testY(end+1) = group_activity_i;
                else
                    testX(end+1, :) = feat_vec(:)';
                    testY(end+1) = group_activity_i;
                end
            end
        end
        t = t + 10;
    end
end
testY = testY';