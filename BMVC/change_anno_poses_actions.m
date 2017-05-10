data_dir = './ActivityDataset';
anno_dir = fullfile(data_dir, 'Final_annotations');
out_dir = fullfile(data_dir, 'Final_annotations_split1');
mkdir(out_dir)
pose_dir = './common\pose';
action_dir = './split1\atomic';

action_vec = csvread(fullfile(action_dir, 'actions.txt'));
action_meta = csvread(fullfile(action_dir, 'meta.txt'));

for seqi = 1:44
    if seqi == 39, continue, end;
    pose_vec = csvread(fullfile(pose_dir, sprintf('pose%2.2d.txt', seqi)));
    pose_meta = csvread(fullfile(pose_dir, sprintf('meta%2.2d.txt', seqi)));
    annofile = fullfile(anno_dir, sprintf('data_%2.2d.mat', seqi));
    outfile = fullfile(out_dir, sprintf('data_%2.2d.mat', seqi));
    anno = load(annofile);
    anno = anno.anno_data;
    anno_data = anno;
    pose_change = zeros(anno.nframe, numel(anno.people));
    action_change = zeros(anno.nframe, numel(anno.people));
    
    for t = 1:anno.nframe
        for nped = 1:numel(anno.people)
            pose = pose_vec(pose_meta(:,2) == t & pose_meta(:,3) == nped, :);
            
            if rem(t-1, 10) == 0 && t > 10
                action = action_vec(action_meta(:,1) == seqi & action_meta(:,2) == t-1 & action_meta(:,3) == nped, :);
                if numel(action) == 20
                    action = reshape(action, [10, 2]);
%                     anno_data.people(nped).action(t-9:t)
%                     [~, anno_data.people(nped).action(t-9:t)] = max(action, [], 2);
%                     anno_data.people(nped).action(t-9:t)
                    action_change(t-9:t, nped) = 1;
                end
            end
            if numel(pose) > 0
%                 anno_data.people(nped).pose(t)
                [~, anno_data.people(nped).pose(t)] = max(pose);
%                 anno_data.people(nped).pose(t)
                
                pose_change(t, nped) = 1;
            end
            
            
        end
    end
    sum(pose_change(:)/numel(pose_change))
    sum(action_change(:)/numel(action_change))
    anno = [];
    anno.anno_data = anno_data;
    save(outfile, 'anno');
end