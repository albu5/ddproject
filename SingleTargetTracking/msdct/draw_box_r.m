function I = draw_box_r(X,Y,hx,hy,I)
for i=X-hx:1:X+hx,
    for j=Y-hy:2*hy:Y+hy,
          I(round(j),round(i),1)=255;
          I(round(j),round(i),2)=0;
          I(round(j),round(i),3)=0;       
    end
end
for i=X-hx:2*hx:X+hx,
    for j=Y-hy:1:Y+hy,
          I(round(j),round(i),1)=255;
          I(round(j),round(i),2)=0;
          I(round(j),round(i),3)=0;          
    end
end

