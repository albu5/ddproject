function I = draw_box(X,Y,hx,hy,I)

for i=X-hx:1:X+hx,
    for j=Y-hy:2*hy:Y+hy,
          I(round(j),round(i),1)=0;
          I(round(j),round(i),2)=0;
          I(round(j),round(i),3)=255;       
    end
end
for i=X-hx:2*hx:X+hx,
    for j=Y-hy:1:Y+hy,
          I(round(j),round(i),1)=0;
          I(round(j),round(i),2)=0;
          I(round(j),round(i),3)=255;          
    end
end

