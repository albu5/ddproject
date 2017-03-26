function showDets(imdir, dets)

imfiles = dir([imdir '*.jpg']);

for i = 1:length(imfiles)
    imshow([imdir imfiles(i).name]);
    
    drawTracks(dets(dets(:,1)==i, :));
    
    drawnow;
end

end



function drawTracks(dets)
cmap = colormap;

for i = 1:size(dets,1)
    col = cmap(mod(i*10, 64) + 1, :);
    rectangle('Position', dets(i,2:5), 'EdgeColor', col, 'LineWidth', 3);
end

end