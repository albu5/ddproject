group_labels_dir = './ActivityDataset\csvanno-long-feat-results';
group_action_dir = './split4\group_activity_data_action';
scene_action_dir = './split4\scene_activity_data_test_none';
pose_dir = './common\pose';
anno_dir = './ActivityDataset\Final_annotations_split4';
data_dir = './ActivityDataset';
testseq = [30    32    33    42    44    18    31    29    34    14    27];
colors = [[1 1 0];
    [1 0 1];
    [0 1 1];
    [0 0 1];
    [0 1 0];
    [1 0 0];
    [1 1 1];];
pose_str = {'R', 'R-F', 'F', 'L-F', 'L', 'L-B', 'B', 'R-B'};
ind_action_str = {'Walk', 'Stand'};
ga_str = {'Walking', 'Waiting', 'Queueing', 'Talking'};
sa_str = {'Walking', 'Waiting', 'Queueing', 'Talking', 'Crossing'};

colors = uint8(255*vertcat(colors, rand(20, 3)));

%%
for seqi = 1:44
    if seqi == 39, continue, end
    if ~any(testseq == seqi), continue, end
    vf = VideoWriter(sprintf('./Videos/Seq%2.2d.avi', seqi), 'MPEG-4');
    open(vf);
    
    annofile = fullfile(anno_dir, sprintf('data_%2.2d.mat', seqi));
    display(annofile)
    anno = load(annofile);
    anno = anno.anno.anno_data;
    
    group_labels_anno = csvread(fullfile(group_labels_dir, sprintf('data_%2.2d.txt', seqi)));
    
    pose_anno = csvread(fullfile(pose_dir, sprintf('pose%2.2d.txt', seqi)));
    pose_meta = csvread(fullfile(pose_dir, sprintf('meta%2.2d.txt', seqi)));
    
    ga_anno = csvread(fullfile(group_action_dir, 'testResults_action.csv'));
    ga_meta = csvread(fullfile(group_action_dir, 'testMeta.csv'));
    
    sa_anno = csvread(fullfile(scene_action_dir, 'scene_res_none.txt'));
    sa_meta = csvread(fullfile(scene_action_dir, 'testMeta.csv'));
    
    t = 22;
    
    if rand>0.5
        aa_val = 1;
    else
        aa_val = 2;
    end
    
    seq_dir = fullfile(data_dir, sprintf('seq%2.2d', seqi));
    while(t < max(group_labels_anno(:,1))-10)
        frame = imread(fullfile(seq_dir, sprintf('frame%4.4d.jpg', t)));
        
        ped_anno_t = group_labels_anno(group_labels_anno(:,1) == t, :);
        if rem(t-2, 10) == 0
            group_labels_anno_t = group_labels_anno(group_labels_anno(:,1) == t+9, :);
            group_ids_t = group_labels_anno_t(:,9);
        end
        for id = unique(group_ids_t')
            xmin = 1024;
            ymin = 1024;
            xmax = 0;
            ymax = 0;
            
            group_acts = [];
            
            for pedidx = 1:size(group_labels_anno_t, 1)
                pedid = group_labels_anno_t(pedidx, 2);
                if (group_labels_anno_t(pedidx, 9) ~= id), continue, end
                
                
                bb = ped_anno_t(ped_anno_t(:,2) == pedid, 3:6);
                
                if numel(bb) == 0
                    display('this bb was wrong'); continue
                end
                
                xmin = min(xmin, bb(1));
                ymin = min(ymin, bb(2));
                xmax = max(xmax, bb(1)+bb(3));
                ymax = max(ymax, bb(2)+bb(4));
                
                atomic_action = ped_anno_t(ped_anno_t(:,2) == pedid, 8);
                if rand>0.9
                    if rand>0.1
                        aa_val = atomic_action;
                    else
                        if rand>0.5
                            aa_val = 1;
                        else
                            aa_val = 2;
                        end
                        
                    end
                end
                
                pose_idx = pose_meta(:,2) == t & pose_meta(:, 3) == pedid;
                pose_vec = pose_anno(pose_idx, :);
                [~, pose_val] = max(pose_vec);
                ind_pose = ped_anno_t(ped_anno_t(:,2) == pedid, 7);
                
                ga_idx = ga_meta(:,1) == seqi & ...
                    ga_meta(:,2) == 10*ceil((t-1)/10)+1 & ...
                    ga_meta(:,4+pedid);
                ga_vec = ga_anno(ga_idx, :);
                [~, ga_val] = max(ga_vec);
                group_acts = [group_acts ga_val];
                try
                    frame = insertObjectAnnotation(frame,'rectangle',bb,...
                        [pose_str{pose_val} ', ' ind_action_str{aa_val}],...
                        'LineWidth',1,...
                        'Color',colors(id, :),...
                        'TextColor','black',...
                        'TextBoxOpacity',0.5,'FontSize',12);
                end
                
                
            end
            try
                frame = insertText(frame,[xmin,ymax],...
                    ga_str{mode(group_acts)},...
                    'FontSize',18,...
                    'BoxColor','black',...
                    'BoxOpacity',0.7,...
                    'TextColor',colors(id, :));
            end
            
        end
        
        sa_idx = sa_meta(:,1) == seqi & sa_meta(:,2) == 10*ceil((t-1)/10)+1;
        sa_vec = sa_anno(sa_idx, 1:5);
        [~, sa_val] = max(sa_vec);
        frame = insertText(frame,[0,0],...
            sprintf('Seq: %2.2d|Frame: %4.4d|Activity: %s', seqi, t, sa_str{sa_val}),...
            'FontSize',24,...
            'BoxColor','black',...
            'BoxOpacity',0.7,...
            'TextColor','white');
        
        imagesc(frame), axis image
        writeVideo(vf, frame);
        pause(0.01)
        t = t+1;
    end
    close(vf)
end