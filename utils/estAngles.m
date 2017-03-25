% function viewAngles(vidfile, trackfile)
vidfile = 'C:\Users\Ashish\Desktop\DDP\agutils\data\road-seg2\road_seg2.avi';
trackfile = 'C:\Users\Ashish\Desktop\DDP\agutils\data\road-seg2\track.txt';
outfile = 'C:\Users\Ashish\Desktop\DDP\agutils\data\road-seg2\road_seg2_vfoa.avi';
v = VideoReader(vidfile);
vo = VideoWriter(outfile);
open(vo);
tracks = csvread(trackfile);
alive_thresh = 5;
eeta = 1;
time_period = 10*1; %frame rate * time in seconds
R0 = 30;
R1 = 100;
frame = round(255*rand(v.Height, v.Width));
% format
% frame, id, bb_left, bb_top, bb_w, bb_h, conf, x3d, y3d, z3d

%%
numid = max(tracks(:,2));
% columns are nframes, alive, theta, omega, r
% rows are ids
trackmat = zeros(numid, 5);

%display
c = round(255*rand(512,3));

%%
% create human struct
human_template.x = [];
human_template.y = [];
for i = 1:numid
    humans{i} = human_template;
end

%%

deti = 0;
framei = 0;
while deti < size(tracks,1)
    deti = deti + 1
    %%
    % estimate angles
    if tracks(deti, 2)>0 % ignore detections without ids
        id = tracks(deti,2);
        trackmat(id,1) = trackmat(id,1) + 1;
        nf = trackmat(id,1);
        if nf > alive_thresh
            trackmat(id,2) = 1;
        end
        %store feet position
        humans{id}.x(end+1) = tracks(deti,3) + round(tracks(deti,5)/2);
        humans{id}.y(end+1) = tracks(deti,4) + round(tracks(deti,6));
        
        %get slope
        if trackmat(id,2)
            if numel(humans{id}.y) <= time_period
                Y = humans{id}.y';
                X = humans{id}.x';              
            else
                Y = humans{id}.y(end-time_period:end)';
                X = humans{id}.x(end-time_period:end)';              
            end
            % VFOA computation goes here
            [bxy,~,rxy] = regress(Y,horzcat(ones(size(X)),X));
            [bx,~,rx] = regress(X,horzcat(ones(size(X)),(1:numel(X))'));
            [by,~,ry] = regress(Y,horzcat(ones(size(Y)),(1:numel(Y))'));
            spd = bx(2)^2 + by(2)^2;
%             rxy = spd;
%             omega = 2*pi*(1-exp(-(mean(rxy.^2))/R1));
%             omega = pi/8+(2*pi-pi/8)*(1-exp(-R0/(mean(rxy.^2))));
            omega = 2*pi*erf(sqrt(mean(rxy.^2))/spd);
            theta =atan(bxy(2));
            quad = atan2(by(2),bx(2));
            if theta*quad < 0
                theta = theta - pi*abs(theta)/theta;
            end
%             r = R1/4 + R1*exp(-(mean(rxy.^2))/R0);
%             r = R1/3 + R1*exp(-R0/(mean(rxy.^2)));
            r = R0 + 2*spd;
            trackmat(id,3) = theta;
            trackmat(id,4) = omega;
            trackmat(id,5) = r;
        end
    end
    if framei < tracks(deti,1)
        framei = framei + 1;
        imagesc(frame), axis image, axis off
        pause(0.001)
        alive_ids = tracks((tracks(:,1) == (framei-1)), 2);
        for alive_id = alive_ids'
            t = linspace(trackmat(alive_id,3)-trackmat(alive_id,4)/2,...
                trackmat(alive_id,3)+trackmat(alive_id,4)/2, 8); % 8 polygon approx
            x = humans{alive_id}.x(end) + trackmat(alive_id,5)*cos(t);
            y = humans{alive_id}.y(end) + trackmat(alive_id,5)*sin(t);
            x = [humans{alive_id}.x(end) x];
            y = [humans{alive_id}.y(end) y];
            hold on
            f = fill(x,y,c(alive_id,:)/255);
            hold off
            alpha(f,0.3)
            wframe = getframe;
            writeVideo(vo,wframe);
        end
        frame = (readFrame(v));
        %annotate frame
        frame = insertObjectAnnotation(frame,'rectangle', ...
            round(tracks(deti,3:6)),...
            num2str(tracks(deti,2)),...
            'Color',c(tracks(deti,2),:),...
            'TextBoxOpacity',0.1,...
            'FontSize',10);
    else
        %annotate frame
        frame = insertObjectAnnotation(frame,'rectangle', ...
            round(tracks(deti,3:6)),...
            num2str(tracks(deti,2)),...
            'Color',c(tracks(deti,2),:),...
            'TextBoxOpacity',0.1,...
            'FontSize',10);
    end
    %%
    % show angles
end
close(vo)
