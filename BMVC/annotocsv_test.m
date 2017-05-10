data_dir = './ActivityDataset';
anno_dir = fullfile(data_dir, 'my_anno');
% exist(anno_dir, 'dir')
pre_anno = 'data_%2.2d';
pre_seq = 'seq%2.2d';
out_dir = fullfile(data_dir, 'csvanno-long-feat-split4-action');
mkdir(out_dir);

action_vec = csvread('split4/atomic/actions.txt');
action_meta = csvread('split4/atomic/meta.txt');

%%
for seqi=1:44
    if seqi == 39, continue, end
    pose_vec = csvread(sprintf('common/pose/pose%2.2d.txt',seqi));
    pose_meta = csvread(sprintf('common/pose/meta%2.2d.txt',seqi));
    
    seqdir = fullfile(data_dir, sprintf(pre_anno, seqi));
    annofile = fullfile(anno_dir, sprintf(pre_anno, seqi));
    anno = load(annofile);
    anno = anno.anno_data;
    data = [];
    for t = 1:anno.nframe
        for n = 1:numel(anno.people)
            if sum(anno.people(n).bbs(t, :)) > 0
                
                start_t = max(1, t-9);
                validt = sum(anno.people(n).bbs, 2) > 0;
                validt = validt & ((1:anno.nframe)' >= start_t) & ((1:anno.nframe)' <= t);
                
%                 if rem(t-1, 10) == 0
                if sum(validt) > 9 && rem(t-1, 10) == 0
                    fbbs = anno.people(n).bbs(validt, :);
                    fbbs(:,1) = fbbs(:,1) + fbbs(:,3)/2;
                    fbbs(:,2) = fbbs(:,2) + fbbs(:,4);
                    
                    idx_pose = pose_meta(:,1) == seqi &...
                        pose_meta(:,2) >= start_t &...
                        pose_meta(:,2) <= t &...
                        pose_meta(:,3) == n;
                    
                    idx_action = action_meta(:,1) == seqi &...
                        action_meta(:,2) >= start_t &...
                        action_meta(:,2) <= t &...
                        action_meta(:,3) == n;
                    
                    pose_prob = pose_vec(idx_pose, :);
                    [~, fpose] = max(mean(pose_prob,1));
                    
                    if sum(idx_action) == 1
                        action_prob = reshape(action_vec(idx_action, :), [10, 2]);
                        [~, faction] = max(mean(action_prob,1));
                    else
                        action_prob = reshape(action_vec(idx_action, :), [sum(idx_action)*10, 2]);
                        [~, faction] = max(mean(action_prob,1));
                    end
                    

                    temp1 = mode(fpose, 1);
                    temp2 = mode(faction, 1);
                    bbf = fbbs;
                    bias = ones(size(bbf,1),1);
                    predictor = [bias, (1:numel(bias))'];
                    [B2,~,R2] = regress(bbf(:,1),predictor);
                    [B3,~,R3] = regress(bbf(:,2),predictor);
                    [B4,~,R4] = regress(bbf(:,3),predictor);
                    [B5,~,R5] = regress(bbf(:,4),predictor);
%                     meanbbs = mean(bbf,1);
                    meanbbs = bbf(end, :);
                    fvec = horzcat(B2(:)', B3(:)',B4(:)',B5(:)', ...
                        mean(R2), mean(R3), mean(R4), mean(R5), ...
                        temp1, temp2, meanbbs);
                    fvec2 = [B2(2), B3(2), B4(2), B5(2), temp1, temp2, meanbbs];
                    [tx, ty] = getTriVertices(meanbbs, fpose, 2);
                    fvec3 = [B2(2), B3(2), B4(2), B5(2),...
                        (1:8)*(temp1'), (1:2)*(temp2'),...
                        meanbbs, ...
                        tx', ty'];
                    fvec4 = [B2(2), B3(2), B4(2), B5(2),...
                        fpose, faction,...
                        meanbbs, ...
                        tx', ty'];
                else
                    fvec = zeros(1, 26);
                    fvec2 = zeros(1, 18);
                    fvec3 = zeros(1, 16);
                    fvec4 = zeros(1, 16);
                end
                if anno.groups.grp_label(t, n) > 0
                    grpact = anno.groups.grp_act(t, anno.groups.grp_label(t, n));
                else
                    grpact = 0;
                end
                row = [t, n, anno.people(n).bbs(t, :),...
                    anno.people(n).pose(t), anno.people(n).action(t), ...
                    anno.groups.grp_label(t, n),...
                    grpact,...
                    fvec4];
                
                if numel(data) == 0
                    data = row;
                else
                    data = vertcat(data, row);
                end
            end
        end
    end
    display(fullfile(out_dir, sprintf([pre_anno '.txt'], seqi)))
    csvwrite(fullfile(out_dir, sprintf([pre_anno  '.txt'], seqi)), data);
end