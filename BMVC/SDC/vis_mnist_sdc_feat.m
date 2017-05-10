x_train = csvread('train_feat.txt');
x_test = csvread('test_feat.txt');
y_train = csvread('train_labels');
y_test = csvread('test_labels');

%%
ydata_train = tsne(x_train(1:2000, :), y_train(1:2000));

%%
ydata_test = tsne(x_test(1:2000, :), y_test(1:2000));

%%
[idx_train, c_train] = kmeans(x_train(1:2000, :), 10);
[idx_test, c_test] = kmeans(x_test(1:2000, :), 10);


x_train_trim = x_train(1:2000, :);
x_test_trim = x_test(1:2000, :);

%%
for i=1:10
   scatter(ydata_train(idx_train == i, 1), ydata_train(idx_train == i, 2)), hold on
end
drawnow()

%%
for i=1:10
   scatter(ydata_test(idx_test == i, 1), ydata_test(idx_test == i, 2)), hold on
end
drawnow()

%%
hung_mat = zeros(10);
for i=1:10
    for j = 1:10
        ci = i;
        yi = j;
        idx = idx_train == ci;
        hung_mat(i,j) = -sum(double(y_train(idx) == yi));
    end
end
[cimap, cost] = munkres(hung_mat);
-cost/2000
