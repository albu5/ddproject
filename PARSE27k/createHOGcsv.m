% trainx = zeros(numel(dir('PARSE27/train/*.jpg')),...
%     numel(extractHOGFeatures(imread('PARSE27/train/1.jpg'))));
% validx = zeros(numel(dir('PARSE27/valid/*.jpg')),...
%     numel(extractHOGFeatures(imread('PARSE27/valid/1.jpg'))));

i = 0;
while true
    i = i+1;
    try
        im3 = imread(sprintf('PARSE27/train/%d.jpg', i));
        hog_vec = extractHOGFeatures(im3);
        dlmwrite('PARSE27/train/hog_vec.txt', hog_vec, '-append')
%         trainx = vertcat(trainx, hog_vec);
        if rem(i,100)==0
            display(sprintf('%d',i));
        end
    catch
        break
    end
end
% csvwrite('PARSE27/train/hog_vec.txt', trainx);

i = 0;
while true
    i = i+1;
    try
        im3 = imread(sprintf('PARSE27/valid/%d.jpg', i));
        hog_vec = extractHOGFeatures(im3);
%         validx = vertcat(validx, hog_vec);
        dlmwrite('PARSE27/valid/hog_vec.txt', hog_vec, '-append')
        if rem(i,100)==0
            display(sprintf('%d',i));
        end
    catch
        break
    end
end
% csvwrite('PARSE27/valid/hog_vec.txt', validx);
