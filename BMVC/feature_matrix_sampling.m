function y = feature_matrix_sampling(group)

N = numel(group);
y = zeros(N);
for i = 1:N
    for j = 1:N
        if group(i).group ~= group(j).group
            y(i,j) = -1;
        end
    end
end


D = numel(zeros);

% closeness criteria over 10 timesteps
for i = 1:N
    for j = 1:N
        bbi = group(i).bbs(end-9:end, :);
        bbj = group(j).bbs(end-9:end, :);
        dij = bbi(:) - bbj(:);
        D(i,j) = (dij'*dij)/(group(i).group ~= group(j).group + 1e-8);
    end
end

[~, I] = sort(D, 2);
nsamp = 2;
group_vec = [];

for i = 1:N
    group_vec(i) = group(i).group;
end

for i = 1:N
   for j = 1:min(nsamp, sum(group_vec == group(i).group))
       y(i, I(i,j)) = 1;
   end
end