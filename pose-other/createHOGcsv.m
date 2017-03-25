trainx = [];
validx = [];

i = 0;
while true
    i = i+1;
    try
        im = imread(sprintf('MULTIVIEW96x40/train/%d.png', i));
        im3 = double(cat(3,im,im,im))/255;
        hog_vec = extractHOGFeatures(im3);
        trainx = vertcat(trainx, hog_vec);
        if rem(i,100)==0
            display(sprintf('%d',i));
        end
    catch
        break
    end
end
csvwrite('MULTIVIEW96x40/train/hog_vec.txt', trainx);

i = 0;
while true
    i = i+1;
    try
        im = imread(sprintf('MULTIVIEW96x40/valid/%d.png', i));
        im3 = double(cat(3,im,im,im))/255;
        hog_vec = extractHOGFeatures(im3);
        validx = vertcat(validx, hog_vec);
        if rem(i,100)==0
            display(sprintf('%d',i));
        end
    catch
        break
    end
end
csvwrite('MULTIVIEW96x40/valid/hog_vec.txt', validx);
