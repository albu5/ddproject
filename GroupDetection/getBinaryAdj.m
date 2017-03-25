function adjM = getBinaryAdj(scene)
tic
N = numel(scene.X);
adjM = zeros(N);
for i = 1:N
    for j = 1:N
        card = 2-double(i==j);
       adjM(i,j) = card*max(max(scene.attention(:,:,i).*scene.attention(:,:,j)));
       adjM(j,i) = adjM(i,j); 
    end
end
toc