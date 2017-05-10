function y = oneHot(x, nclasses)

x(x<1) = 1;
y = zeros(numel(x), nclasses);
for i = 1:numel(x)
   y(i,x(i)) = 1; 
end