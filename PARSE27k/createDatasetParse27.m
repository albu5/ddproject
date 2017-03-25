load crops/crops.mat

badtest = (test.labels(2,:)==0);
badtrain = (train.labels(2,:)==0);
badvalid = (valid.labels(2,:)==0);

sum(badtest)
sum(badtrain)
sum(badvalid)

test.labels = test.labels(:,~badtest);
valid.labels = valid.labels(:,~badvalid);
train.labels = train.labels(:,~badtrain);
test.images = test.images(:,:,:,~badtest);
valid.images = valid.images(:,:,:,~badvalid);
train.images = train.images(:,:,:,~badtrain);

%%
N = size(test.labels, 2);
mkdir test
cd test
for i=1:N
    % if test.labels(2,i) == 0, continue, end
    imwrite(test.images(:,:,:,i), sprintf('%d.jpg', i));
    if rem(i,100)==0
        display(sprintf('%d out of %d',i,N));
    end
end
labels=test.labels(2,:)';
labels4=test.labels(1,:)';
csvwrite('labels.txt', labels);
csvwrite('labels4.txt', labels4);
cd ..


%%
N = size(train.labels, 2);
mkdir train
cd train
for i=1:N
    imwrite(train.images(:,:,:,i), sprintf('%d.jpg', i));
    if rem(i,100)==0
        display(sprintf('%d out of %d',i,N));
    end
end
labels=train.labels(2,:)';
labels4=train.labels(1,:)';
csvwrite('labels.txt', labels);
csvwrite('labels4.txt', labels4);
cd ..


%%
N = size(valid.labels, 2);
mkdir valid
cd valid
for i=1:N
    imwrite(valid.images(:,:,:,i), sprintf('%d.jpg', i));
    if rem(i,100)==0
        display(sprintf('%d out of %d',i,N));
    end
end
labels=valid.labels(2,:)';
labels4=valid.labels(1,:)';
csvwrite('labels.txt', labels);
csvwrite('labels4.txt', labels4);
cd ..
