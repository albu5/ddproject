function [y_in, y_out] = find_bad_examples(res_mat, group)

N = numel(group);
 
y_in = zeros(N,1);
y_out = zeros(N,1);

true_mat = zeros(N);
for i = 1:N
    for j = 1:N
        true_mat(i, j) = group(i).group == group(j).group;
    end
end

for i = 1:N
   out_violators = (~true_mat(i, :)) .* res_mat(i, :);
   if sum(out_violators) > 0
       y_out(i) = 1;
   end
end

for i = 1:N
    in_violators = true_mat(i, :) .* res_mat(i, :);
    if sum(in_violators) < 1
        y_in(i) = 1;
    end
end