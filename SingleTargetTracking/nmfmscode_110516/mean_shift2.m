function [region,bhat_coeff,p_u,w] = mean_shift2(I,state,region,bins)
binwidth=round(256/bins);
% trial1=zeros(frames,20);
% trial2=zeros(frames,20);
q_u = state.q_u1;
for iter=1:1:20
    
    true=0;
    
    h_x = (region(3)-1)/2;
    h_y = (region(4)-1)/2;
    center = [region(1)+h_x, region(2)+h_y];
    x1 = center(1)-h_x;
    y1 = center(2)-h_y;
    x2 = center(1)+h_x;
    y2 = center(2)+h_y;
    
    p_u = hist_model(I,bins,center,h_x,h_y);
    
    [sizex,sizey,~] = size(I);
    
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
                
                
                if(p_u(r,g,b)==0.0)
                    weight=0;
                else
                    weight=sqrt(q_u(r,g,b)/p_u(r,g,b));
                    % temp4(i,j)=weight;
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

region = [oldcenter(1)-h_x, oldcenter(2)-h_y, 2*h_x+1, 2*h_y+1];
bhat_coeff= bhattacharya_coeff(q_u,p_u,bins);
w=weight;
return