files = dir('PARSE27/train/*.jpg');
for file = files'
    try
        fname = file.name;
        I = imread(sprintf('PARSE224/train/%s', fname));
%         I = imcrop(I, [128/2-72/2, 192/2-144/2, 72-1, 144-1]);
        I = imresize(I,[224,224]);
        imwrite(I,sprintf('PARSE224/train/%s', fname));
    catch
        break
    end
end

files = dir('PARSE27/valid/*.jpg');
for file = files'
    try
        fname = file.name;
        I = imread(sprintf('PARSE224/valid/%s', fname));
%         I = imcrop(I, [128/2-72/2, 192/2-144/2, 72-1, 144-1]);
        I = imresize(I,[224,224]);
        imwrite(I,sprintf('PARSE224/valid/%s', fname));
    catch
        break
    end
end


files = dir('PARSE27/test/*.jpg');
for file = files'
    try
        fname = file.name;
        I = imread(sprintf('PARSE224/test/%s', fname));
%         I = imcrop(I, [128/2-72/2, 192/2-144/2, 72-1, 144-1]);
        I = imresize(I,[224,224]);
        imwrite(I,sprintf('PARSE224/test/%s', fname));
    catch
        break
    end
end

