function OverlapArea = getOverlap(bb1, pose1, bb2, pose2, range, debug)
try
    display(debug);
catch
    debug = false;
end

[x1, y1] = getTriVertices(bb1, pose1, range);
[x2, y2] = getTriVertices(bb2, pose2, range);

[xi, yi] = polybool('intersection', x1, y1, x2, y2);
if debug
    plot([x1; x1(1)],[y1; y1(1)],'r')
    hold on
    plot([x2; x2(1)],[y2; y2(1)],'g')
    fill(xi, yi, 'b', 'faceAlpha', '0.3')
    hold off, axis image
end
OverlapArea = polyarea(xi, yi);