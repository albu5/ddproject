function y = feature_matrix(group)

N = numel(group);

for i = 1:N
    for j = 1:N
        y{i,j} = pairwise_features(group(i).features, group(j).features);
    end
end
        