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

