function msdct
% ncc VOT integration example
%
% This function is an example of tracker integration into the toolkit.
% The implemented tracker is a very simple NCC tracker that is also used as
% the baseline tracker for challenge entries.
%

% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
cleanup = onCleanup(@() exit() );

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('rectangle');

% debugging********************************
% imshow(imread(image))
% hold on
% rectangle('Position', region,...
% 	'EdgeColor','r', 'LineWidth', 3)
% pause(1)
% save C:\Users\Ashish\Desktop\vot\dump\region.mat region
% *****************************************

% prepare log
nowtime = clock;
mylog = load('C:\Users\Ashish\Desktop\vot2015\dump\mylog.mat');
mylog = mylog.mylog;
mylog{end+1} = '************';
mylog{end+1} = [date ' ' num2str(nowtime(4)) ':' num2str(nowtime(5)) ':' num2str(round(nowtime(6)))];
save('C:\Users\Ashish\Desktop\vot2015\dump\mylog.mat', 'mylog');

% Initialize the tracker*******************
mylog{end+1} = 'Initializing tracker ...';
save('C:\Users\Ashish\Desktop\vot2015\dump\mylog.mat', 'mylog');

h_x = (region(3)-1)/2;
h_y = (region(4)-1)/2;
center = [region(1)+h_x, region(2)+h_y];
I = 255*im2double(imread(image));%covert rgb image to double format
params.bins = 16;


mylog{end+1} = 'Initializing state ...';
save('C:\Users\Ashish\Desktop\vot2015\dump\mylog.mat', 'mylog');

state.q_u=hist_model(I,params.bins,center,h_x,h_y);
state.init = false;
state.Iprev = I;
% *****************************************

while true
    
    % **********************************
    % VOT: Get next frame
    % **********************************
    [handle, image] = handle.frame(handle);
    
    if isempty(image)
        break;
    end;
    
    % Perform a tracking step, obtain new region
    h_x = (region(3)-1)/2;
    h_y = (region(4)-1)/2;
    center = [region(1)+h_x, region(2)+h_y];
    I = 255*im2double(imread(image));
    
    if ~state.init
        [state.dctwt_q,~,~]=dct_wt3(I,state.Iprev,center,h_x,h_y);
        state.init = true;
        state.Iprev = I;
        mylog{end+1} = 'Initializing tracker completed...';
        save('C:\Users\Ashish\Desktop\vot2015\dump\mylog.mat', 'mylog');
    else
        
        mylog{end+1} = 'Running iteration ...';
        save('C:\Users\Ashish\Desktop\vot2015\dump\mylog.mat', 'mylog');
        
        [state.dctwt_p,~,~]=dct_wt3(I,state.Iprev,center,h_x,h_y);
        [center,~,state.p_u,~] = mean_shift_dct2(I,center,state.q_u,h_x,h_y,params.bins,state.dctwt_q,state.dctwt_p);
        region = [center(1)-h_x, center(2)-h_y, 2*h_x+1, 2*h_y+1];
        
        mylog{end+1} = 'Running iteration completed ...';
        save('C:\Users\Ashish\Desktop\vot2015\dump\mylog.mat', 'mylog');
        
    end
    
    % **********************************
    % VOT: Report position for frame
    % **********************************
    handle = handle.report(handle, region);
    
end;

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle);

end

function [ dctwt,pres_frbg,prev_frbg ] = dct_wt3(I,I_prev,center,h_x,h_y)
%dct_wt3 calculates weight to achieve illumination invariance
% dctwt=(fr/bg)prev/(fr/bg)pres

%Spatial Ratio(present frame)
I1=rgb2ycbcr(I);
h_xb=2*h_x;
h_yb=2*h_y;
Y1=I1(:,:,1);
cb1=I1(:,:,2);
cr1=I1(:,:,3);

[Y1_fr]=crop_image(Y1,center,h_x,h_y);
[Y1_bg]=crop_image(Y1,center,h_xb,h_yb);

Y1_log=log(Y1);
Y1_dct=dct2(Y1_log);
Y1_dctmx=max(Y1_dct(:));
Y1_dct=Y1_dct/max(Y1_dct(:));

Y1_fr_log=log(Y1_fr);
Y1_fr_dct=dct2(Y1_fr_log);
Y1_fr_norm=Y1_fr_dct/Y1_dctmx;

Y1_bg_log=log(Y1_bg);
Y1_bg_dct=dct2(Y1_bg_log);
Y1_bg_norm=Y1_bg_dct/Y1_dctmx;

dc_Y1=Y1_dct(1,1);
dc_fr_Y1=Y1_fr_norm(1,1);
dc_bg_Y1=Y1_bg_norm(1,1);

dc_fr_Y1_norm=dc_fr_Y1/dc_Y1;
dc_bg_Y1_norm=dc_bg_Y1/dc_Y1;

pres_frbg=dc_fr_Y1_norm/dc_bg_Y1_norm;


%Spatial Ratio(previous frame)
I2=rgb2ycbcr(I_prev);
h_xb=2*h_x;
h_yb=2*h_y;
Y2=I2(:,:,1);
cb2=I2(:,:,2);
cr2=I2(:,:,3);

[Y2_fr]=crop_image(Y2,center,h_x,h_y);
[Y2_bg]=crop_image(Y2,center,h_xb,h_yb);

Y2_log=log(Y2);
Y2_dct=dct2(Y2_log);
Y2_dctmx=max(Y2_dct(:));
Y2_dct=Y2_dct/max(Y2_dct(:));

Y2_fr_log=log(Y2_fr);
Y2_fr_dct=dct2(Y2_fr_log);
Y2_fr_norm=Y2_fr_dct/Y2_dctmx;
dc_fr_pres=Y2_fr_norm(1,1);

Y2_bg_log=log(Y2_bg);
Y2_bg_dct=dct2(Y2_bg_log);
Y2_bg_norm=Y2_bg_dct/Y2_dctmx;

dc_Y2=Y2_dct(1,1);
dc_fr_Y2=Y2_fr_norm(1,1);
dc_bg_Y2=Y2_bg_norm(1,1);

dc_fr_Y2_norm=dc_fr_Y2/dc_Y2;
dc_bg_Y2_norm=dc_bg_Y2/dc_Y2;

prev_frbg=dc_fr_Y2_norm/dc_bg_Y2_norm;

%Temporal ratio
dctwt=prev_frbg/pres_frbg;

end

function[new_I, new1]=crop_image(I,center,h_x,h_y)
[sizex,sizey,sizez]=size(I);
%new_I=zeros(size(I));
x1 = round(center(1)-h_x);
y1 = round(center(2)-h_y);
x2 = round(center(1)+h_x);
y2 = round(center(2)+h_y);
% [x1 x2 y1 y2 center]

for i=x1:1:x2
    for j= y1:1:y2
        if(j>0 && j<=sizex && i>0 && i<=sizey)
            new_I(j-(y1-1),i-(x1-1),:)=I(j,i,:);
        end
    end
end
% [sizex sizey]=size(I);
for i=1:1:sizey
    for j=1:1:sizex
        if(j>y1 && j<=y2 && i>x1 && i<=x2)
            new1(j,i,:)=new_I(j-(y1-1),i-(x1-1),:);
        else
            new1(j,i,:)=255;
        end
    end
end
end

function model = hist_model(I,bins,center,h_x,h_y)
x1 = center(1)-h_x;
y1 = center(2)-h_y;
x2 = center(1)+h_x;
y2 = center(2)+h_y;
c=0;
binwidth=round(256/bins);
model= zeros(bins,bins,bins);
[sizex,sizey,sizez]=size(I);
for i=x1:1:x2,
    for j=y1:1:y2,
        tempx = (i-center(1))/h_x;
        tempy = (j-center(2))/h_y;
        
        dist =sqrt(tempx^2+tempy^2);
        
        if(dist>1)
            k=0;
        else
            k=1-dist;
        end
        
        if(j>0 && j<=sizex && i>0 && i<=sizey)
            
            r=floor(I(round(j),round(i),1)/binwidth)+1;
            g=floor(I(round(j),round(i),2)/binwidth)+1;
            b=floor(I(round(j),round(i),3)/binwidth)+1;
            
            model(r,g,b)=model(r,g,b)+k*k;
            c=c+k*k;
        end
    end
end
if( c~=0)
    model=model/c;
end
end

function [oldcenter,bhat_coeff,p_u,w] = mean_shift_dct2(I,center,q_u,h_x,h_y,bins,dctwt_q,dctwt_p)
binwidth=round(256/bins);
q_uw=dctwt_q*q_u;


for iter=1:1:20
    
    true=0;
    
    x1 = center(1)-h_x;
    y1 = center(2)-h_y;
    x2 = center(1)+h_x;
    y2 = center(2)+h_y;
    
    p_u = hist_model(I,bins,center,h_x,h_y);
    p_uw= p_u*dctwt_p;
    
    [sizex,sizey,sizez] = size(I);
    
    oldcenter = center;
    
    sumt=0;
    sumx=0;
    sumy=0;
    
    for i=x1:1:x2,
        for j=y1:1:y2,
            
            tempx = (i-oldcenter(1))/h_x;
            tempy = (j-oldcenter(2))/h_y;
            
            if(j>0 && j<=sizex && i>0 && i<=sizey)
                
                true=true+1;
                r=floor(I(round(j),round(i),1)/binwidth)+1;
                g=floor(I(round(j),round(i),2)/binwidth)+1;
                b=floor(I(round(j),round(i),3)/binwidth)+1;
                
                
                if(p_uw(r,g,b)==0.0)
                    weight=0;
                else
                    weight=sqrt((q_uw(r,g,b)/p_uw(r,g,b)));
                end
            else
                weight=0;
            end
            sumt=sumt+weight;
            sumx=sumx+weight*tempx;
            sumy=sumy+weight*tempy;
        end
    end
    
    if(sumt ~= 0)
        sumx=sumx/sumt;
        sumy=sumy/sumt;
    end
    
    temx=oldcenter(1);
    temy=oldcenter(2);
    
    oldcenter(1)=oldcenter(1)+round(sumx*h_x);
    oldcenter(2)=oldcenter(2)+round(sumy*h_y);
    %  x_cor=oldcenter(1);
    %   trial1(frameint,iter)=x_cor;
    %   trial2(frameint,iter)=oldcenter(2);
    if((oldcenter(1)-temx)^2+(oldcenter(2)-temy)^2)<1
        break;
    else
    end
    center=round(oldcenter);
end

bhat_coeff = bhattacharya_coeff(q_uw,p_uw,bins);

w=weight;


end



