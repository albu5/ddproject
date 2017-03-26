function vid1 = readVolume(fvid1)
vid1 = [];
i = 1;
while true
   try
       vid1(:,:,:,i) = double(imread(fullfile(fvid1,sprintf('%d.jpg',i))))/255;
       i = i+1;
   catch 
       break
   end
end
end