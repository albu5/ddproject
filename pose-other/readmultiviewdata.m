datadir = '..\..\..\data\UNPROCESSED\cvpr10_multiview_pedestrians';

addpath('..\..\..\external\xmlio');
validx = [];
validy = [];
%%
model = [100 40 3];

% annotations1 = xml_read([datadir filesep 'viewpoints_train1.al']);
% annotations2 = xml_read([datadir filesep 'viewpoints_train2.al']);
% annotations3 = xml_read([datadir filesep 'viewpoints_train3.al']);
% annotations4 = xml_read([datadir filesep 'viewpoints_train4.al']);
% annotations5 = xml_read([datadir filesep 'viewpoints_train5.al']);
% annotations6 = xml_read([datadir filesep 'viewpoints_train6.al']);
% annotations7 = xml_read([datadir filesep 'viewpoints_train7.al']);
% annotations8 = xml_read([datadir filesep 'viewpoints_train8.al']);
annotations8 = xml_read([datadir filesep 'viewpoints_validate.al']);
%%

idx = 0;
for ann = annotations8.annotation';
    idx = idx + 1;
    image = imread(fullfile(datadir, ann.image.name));
    if ~(size(image,3) == 3)
        image = cat(3,image, image, image);
    end
    
    for annorect = ann.annorect'
        if ~isempty(annorect.silhouette)
            bbh = annorect.y2-annorect.y1;
            bbw = annorect.x2-annorect.x1;
            scale = max(bbh/model(1), bbw/model(2));
            tempim = imresize(image,1/scale);
            midx = floor((annorect.x2+annorect.x1)/(2*scale));
            midy = floor((annorect.y2+annorect.y1)/(2*scale));
            x = midx-model(2)/2+1:midx+model(2)/2;
            y = midy-model(1)/2+1:midy+model(1)/2;
            xprepad = 0;
            yprepad = 0;
            xpostpad = 0;
            ypostpad = 0;
            
            if midx-model(2)/2+1<1
                x = x(x>=1);
                xprepad = -midx+model(2)/2;
            end
            if midx+model(2)/2 > size(tempim,2)
                x = x(x<=size(tempim,2));
                xpostpad = midx+model(2)/2 - size(tempim,2);
            end
            
            if midy-model(1)/2+1<1
                y = y(y>=1);
                yprepad = -midy + model(1)/2;
            end
            if midy+model(1)/2 > size(tempim,1)
                y = y(y<=size(tempim,1));
                ypostpad = midy+model(1)/2 - size(tempim,1);
            end
            
            croppedim = tempim(y,x);
            if xprepad>0
                croppedim = padarray(croppedim, [0 xprepad 0], 'replicate', 'pre');
            end
            if xpostpad>0
                croppedim = padarray(croppedim, [0 xpostpad 0], 'replicate', 'post');
            end
            if yprepad>0
                croppedim = padarray(croppedim, [yprepad 0 0], 'replicate', 'pre');
            end
            if ypostpad>0
                croppedim = padarray(croppedim, [ypostpad 0 0], 'replicate', 'post');
            end
            
            imagesc(croppedim), axis image, colormap gray
            pause(0.0003)
            validx = vertcat(validx, reshape(croppedim,[1 numel(croppedim)]));
            validy = vertcat(validy, annorect.silhouette.id);
        end
    end
    
end