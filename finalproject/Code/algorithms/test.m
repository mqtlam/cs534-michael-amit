clear;
load('../Dataset/Imgvideo0129.mat');

NUM_ENSEMBLES = 10;

X = dataImg(:, [2:8, 10]);
y = dataImg(:, 11);
y(y == 2) = 0;

h1 = learnAdaBoost(X, y, NUM_ENSEMBLES);
y1 = inferAdaBoost(X, h1);
e1 = sum(y ~= y1)/numel(y);
fprintf('AdaBoost training error: %f\n', e1);

h2 = learnDecisionTree(X, y);
y2 = inferDecisionTree(X, h2);
e2 = sum(y ~= y2)/numel(y);
fprintf('Decision tree training error: %f\n', e2);
