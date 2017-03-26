function [jx,jy] = estJitter3(I1,I2,bbs1,bbs2)

pts1 = detectSURFFeatures(I1,'MetricThreshold',500);
pts2 = detectSURFFeatures(I2,'MetricThreshold',500);
tol = 2;
badidx1 = [];
badidx2 = [];

for i = 1:size(pts1,1)
   loc = pts1.Location(i,:);
   for bb = bbs1
      if (loc(1)<bb(3)+tol) && (loc(1)>bb(1)-tol)
          if (loc(2)>bb(2)-tol) && (loc(2)<bb(4)+tol)
              badidx1(end+1) = i;
          end
      end
   end
end
for i = 1:size(pts2,2)
   loc = pts2.Location(i,:);
   for bb = bbs2
      if (loc(1)<bb(3)+tol) && (loc(1)>bb(1)-tol)
          if (loc(2)>bb(2)-tol) && (loc(2)<bb(4)+tol)
              badidx2(end+1) = i;
              
          end
      end
   end
end

pts1(badidx1) = [];
pts2(badidx2) = [];

[fts1,valPts1] = extractFeatures(I1,pts1);
[fts2,valPts2] = extractFeatures(I2,pts2);

%%
idx_pairs = matchFeatures(fts1,fts2);
matchedPts1 = valPts1(idx_pairs(:,1));
matchedPts2 = valPts2(idx_pairs(:,2));

%%
% figure;
% showMatchedFeatures(I1,I2,matchedPts1,matchedPts2);
% title('Matched SURF points,including outliers');

[tform,inPts2,inPts1] = ...
    estimateGeometricTransform(matchedPts2,matchedPts1,...
    'similarity');
jx = tform.T(3,1);
jy = tform.T(3,2);
%%
% figure;
% 
% 
% showMatchedFeatures(I1,I2,inPts1,inPts2);
% title('Matched inlier points');
% 
% outputView = imref2d(size(I1));
% Ir = imwarp(I2,tform,'OutputView',outputView);
% figure; imshow(Ir);
% title('Recovered image');