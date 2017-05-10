M = ((rand(10*10)));
tic
for i = 1:20
    [U,V] = eig(M);
end
toc
