%function viewTrack(vidfile, trackfile)


clear
sequence = 'tudstadtmitte';

calibfile = ['..\..\data\' sequence '\TUD-Stadtmitte-calib.xml'];
vidfile = ['..\..\data\' sequence '\' sequence '.avi'];
trackfile = ['..\..\data\' sequence '\gt.txt'];
outdir = ['..\..\data\' sequence '\pose\'];
addpath(genpath('..\..\external\motutils\camera\'));

sceneInfo.camPar = parseCameraParameters(calibfile);

if ~exist(outdir, 'dir')
    mkdir(outdir);
    mkdir([outdir '0\'])
    mkdir([outdir '45\'])
    mkdir([outdir '90\'])
    mkdir([outdir '135\'])
    mkdir([outdir '180\'])
    mkdir([outdir '225\'])
    mkdir([outdir '270\'])
    mkdir([outdir '315\'])
end

v = VideoReader(vidfile);
tracks = csvread(trackfile);
c = round(256*rand(512,3));
cdata = [];
framewin = 6;
rthresh = inf;
vthresh = 5000;
ynorm = 1;
% format
% frame, id, bb_left, bb_top, bb_w, bb_h, conf, x3d, y3d, z3d

%%

for i = max(tracks(:,2))
    pid(i).pos = [];
    pid(i).size = [];
    pid(i).frameid = [];
end
for i = 1:size(tracks,1)
    frameid = tracks(i,1);
    id = tracks(i,2);
    x = tracks(i,3);
    y = tracks(i,4);
    w = tracks(i,5);
    h = tracks(i,7);
    [Xp,Yp] = projectToGroundPlane((x+w/2), y+h, sceneInfo);
    pid(id).pos(end+1,:) = [Xp,Yp];
    pid(id).size(end+1,1) = w*h;
    pid(id).frameid(end+1,1) = frameid;
end

%%
idx = 0;
res = [];
vx = [];
vy = [];

theta = [];
while hasFrame(v)
    idx = idx + 1;
    I = readFrame(v);
    for i = 1:numel(pid)
        if (sum((pid(i).frameid(:) == idx)) == 1)&&(sum((pid(i).frameid(:) == idx-framewin)) == 1)&&(sum((pid(i).frameid(:) == idx+framewin)) == 1)
            ptr = find(pid(i).frameid(:) == idx);
            X = pid(i).pos(ptr-framewin:ptr+framewin,1);
            Y = pid(i).pos(ptr-framewin:ptr+framewin,2);
            T = ptr-framewin:ptr+framewin;
            T = T(:);
            [Bx,~,Rx] = regress(X, [ones(size(T)) T]);
            [By,~,Ry] = regress(Y, [ones(size(T)) T]);
            vx(end+1) = Bx(2);
            vy(end+1) = ynorm*By(2);
            %            plot3(X,Y,T),title(sprintf('pid: %f', Rx'*Rx+Ry'*Ry));
            %            hold on
            %            plot3(Bx(1)+Bx(2)*T, By(1)+By(2)*T, T)
            %            hold off
            res(end+1) = Rx'*Rx+Ry'*Ry;
            if (vx(end)^2 + vy(end)^2>vthresh)
                if res(end)<rthresh
                    currtrack = tracks((tracks(:,1) == idx)&(tracks(:,2) == i), :);
                    [Iw, theta(end+1)] = writePose(currtrack, I, Bx, By, outdir, sequence);
%                     imagesc(Iw), axis image, title(sprintf('%d pi/8 res= %f', round(8*theta(end)/pi), res(end))), pause(0.03)
                end
            end
            display(sprintf('%f percent completed',100*idx/max(tracks(:,1))))
        end
    end
end


