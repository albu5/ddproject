
function model = hist_model2(I,bins,region)
h_x = (region(3)-1)/2;
h_y = (region(4)-1)/2;
center = [region(1)+h_x, region(2)+h_y];
x1 = center(1)-h_x;
y1 = center(2)-h_y;
x2 = center(1)+h_x;
y2 = center(2)+h_y;

c=0;
binwidth=round(256/bins);
model= zeros(bins,bins,bins);
[sizex,sizey,~]=size(I);
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
return