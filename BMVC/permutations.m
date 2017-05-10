function y = permutations(x)

N = size(x,1);
idx = perms(1:N)';
y = [];

for id = idx
   if numel(y) == 0
       y = x(id, :);
   else
       y = vertcat(y, x(id, :));
   end
end