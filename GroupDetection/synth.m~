scene.dim = [512, 512];
scene.numPed = 10;
R0 = 80;
%%
scene.X = randi(scene.dim(2), [scene.numPed,1]);
scene.Y = randi(scene.dim(1), [scene.numPed,1]);
scene.dir = randi(8,[scene.numPed,1]);
scene.attention = zeros(scene.dim(1), scene.dim(2), scene.numPed);
%%
[idx,idy] = meshgrid(1:scene.dim(1),1:scene.dim(2));
close all
allPed = zeros(scene.dim);
for i = 1:scene.numPed
    x = scene.X(i);
    y = scene.Y(i);
    X = idx-x;
    Y = idy-y;
    R = X.^2 + Y.^2;
    T = atan2d(Y,X);
    dir = (scene.dir(i)-4)*(360/8);
    temp = cosd(T-dir);
    temp(temp<0.3827) = 0;
    temp(temp>0.9239) = 1;
    temp(temp<0.9239 & temp>0.3827) = 0.3827;
    scene.attention(:,:,i) = temp;
    scene.attention(:,:,i) = scene.attention(:,:,i).*(sqrt(R).*((R0^2 - R).*((R0^2 - R)>0)));
    scene.attention(:,:,i) = scene.attention(:,:,i)/(R0^2);
    allPed = allPed + scene.attention(:,:,i);
    imagesc(scene.attention(:,:,i)), axis image, colormap gray
    pause(0.01)
end
imagesc(allPed), axis image, colormap gray

[clusters, adjM] = getClusters(scene)
numel(unique(clusters))