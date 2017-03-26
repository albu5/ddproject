%%
datadir = '../../data/ActivityDataset';
sequences = dir([datadir, filesep, 'seq*']);
delt = 25;
outdir = [datadir filesep 'Trajectories'];

%%
if(~exist(outdir, 'dir'))
    mkdir(outdir)
end

%%
trainX = [];
trainY = [];
testX = [];
testY = [];

testseq = [1     4     5     6     8     2     7    28    35    11    10    26];

for idx = 1:44
    if (sum(testseq==idx)>0), continue, end
    load(fullfile(datadir, 'my_anno', sprintf('data_%2.2d.mat', idx)));
    display(num2str(idx))
    nped = 0;
    for ped = anno_data.people
        nped = nped + 1;
        bbs = ped.bbs;
        validt = sum(bbs,4)>0;
        t = 1;
        while true
            if t+delt>size(bbs,1), break, end
            if sum(validt(t:t+delt-1))==delt
                bbf = ped.bbs(t:t+delt-1,:);
                bias = ones(size(bbf,1),1);
                X = [bias, (1:delt)'];
                [B2,~,R2] = regress(bbf(:,1),X);
                [B3,~,R3] = regress(bbf(:,2),X);
                [B4,~,R4] = regress(bbf(:,3),X);
                [B5,~,R5] = regress(bbf(:,4),X);
                
                pf = ped.pose(t:t+delt-1);
                af = ped.action(t:t+delt-1);
                fvec = horzcat(B2(:)', B3(:)',B4(:)',B5(:)',...
                               mean(R2), mean(R3), mean(R4), mean(R5), ...
                               pf(:)', 8-pf(:)');
                label = mode(af);
                if label == 0, t = t+1;continue;end
                trainX = vertcat(trainX,fvec);
                trainY = vertcat(trainY,label);
                t = t + ceil(sqrt(label)*delt/2);
            else
                t = t+ceil(delt/4);
            end
        end
    end
end


%%
for idx = 1:44
    if (sum(testseq==idx)==0), continue, end
    load(fullfile(datadir, 'my_anno', sprintf('data_%2.2d.mat', idx)));
    display(num2str(idx))
    nped = 0;
    for ped = anno_data.people
        nped = nped + 1;
        bbs = ped.bbs;
        validt = sum(bbs,4)>0;
        t = 1;
        while true
            if t+delt>size(bbs,1), break, end
            if sum(validt(t:t+delt-1))==delt
                bbf = ped.bbs(t:t+delt-1,:);
                bias = ones(size(bbf,1),1);
                X = [bias, (1:delt)'];
                [B2,~,R2] = regress(bbf(:,1),X);
                [B3,~,R3] = regress(bbf(:,2),X);
                [B4,~,R4] = regress(bbf(:,3),X);
                [B5,~,R5] = regress(bbf(:,4),X);
                
                pf = ped.pose(t:t+delt-1);
                af = ped.action(t:t+delt-1);
                fvec = horzcat(B2(:)', B3(:)',B4(:)',B5(:)',...
                               mean(R2), mean(R3), mean(R4), mean(R5), ...
                               pf(:)', 8-pf(:)');
                label = mode(af);
                if label == 0, t = t+1;continue;end
                testX = vertcat(testX,fvec);
                testY = vertcat(testY,label);
                t = t + ceil(sqrt(label)*delt/2);
            else
                t = t+ceil(delt/4);
            end
        end
    end
end

%%
ClassTreeEns = fitensemble(trainX,trainY,'AdaBoostM1',250,'Tree');
yfit = predict(ClassTreeEns,testX);
plotconfusion(ind2vec(testY'), ind2vec(yfit'));
