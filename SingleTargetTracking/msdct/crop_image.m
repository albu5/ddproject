function[new_I new1]=crop_image(I,center,h_x,h_y)

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