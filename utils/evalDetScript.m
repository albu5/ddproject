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

Q = [];

MR = [];


for i = 1:4
[Q(i).kat, MR(i).kat] = evalDet(['../../data/' seqs{i} '/detkat.txt'],...
    ['../../data/' seqs{i} '/det.txt']);

[Q(i).our, MR(i).our] = evalDet(['../../data/' seqs{i} '/detour.txt'],...
    ['../../data/' seqs{i} '/det.txt']);
end

%%
for i = 1:4
    [mean(Q(i).kat(1:end-1)), mean(Q(i).our(1:end-1))]
end
