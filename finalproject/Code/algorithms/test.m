clear;

%% settings
ENSEMBLE_SIZES = [5, 10, 15, 20, 25, 30];
TREE_MAX_DEPTHS = [2, 3, 4, 5, 6, 7];

PERCENT_TRAINING = .75; % percentage of dataset allocated for training
RANDOMIZE_DATASET = 1;  % randomize dataset for training/test split
FIX_SEED = 1;           % fix random seed or not

if FIX_SEED
    rng(12345); % fix seed
end

%% load data
%load('../ProjectDataset/game2/dataset2.mat');
load('../ProjectDataset/game4/dataset4.mat');

% features
X = pData(:, 1:end-1);
[nExamples, nFeatures] = size(X);

% ground truth
y = pData(:, end);
y(y == 2) = 0;

% split into training and testing sets
nTraining = ceil(PERCENT_TRAINING*nExamples);
nTest = nExamples - nTraining;
if RANDOMIZE_DATASET
    augData = shuffle_rows([X, y]);
    X = augData(:, 1:end-1);
    y = augData(:, end);
end
XTrain = X(1:nTraining, :);
yTrain = y(1:nTraining);
XTest = X(nTraining+1:end, :);
yTest = y(nTraining+1:end);

%% evaluate using AdaBoost on different ensemble sizes
adaBoostResults = cell(length(ENSEMBLE_SIZES), 1);
adaBoostTrainErrors = zeros(length(ENSEMBLE_SIZES), 1);
adaBoostTestErrors = zeros(length(ENSEMBLE_SIZES), 1);

cnt = 1;
for size = ENSEMBLE_SIZES
    fprintf('AdaBoost ensemble size: %d\n', size);
    
    h = learnAdaBoost(XTrain, yTrain, size);
    yTrainPred = inferAdaBoost(XTrain, h);
    yTestPred = inferAdaBoost(XTest, h);
    
    eTrain = sum(yTrain ~= yTrainPred)/numel(yTrain);
    aTrain = 1-eTrain;
    fprintf('\tAdaBoost training error: %f\n', eTrain);
    
    eTest = sum(yTest ~= yTestPred)/numel(yTest);
    aTest = 1-eTest;
    fprintf('\tAdaBoost test error: %f\n', eTest);
    
    adaBoostResults{cnt}.ensembleSize = size;
    adaBoostResults{cnt}.trainAccuracy = aTrain;
    adaBoostResults{cnt}.trainConfusionMatrix = confusionmatrix(yTrain, yTrainPred);
    adaBoostResults{cnt}.testAccuracy = aTest;
    adaBoostResults{cnt}.testConfusionMatrix = confusionmatrix(yTest, yTestPred);
    
    adaBoostTrainErrors(cnt) = eTrain;
    adaBoostTestErrors(cnt) = eTest;
    
    cnt = cnt+1;
end

%% evaluate using decision tree
decisionTreeResults = cell(length(TREE_MAX_DEPTHS), 1);
decisionTreeTrainErrors = zeros(length(TREE_MAX_DEPTHS), 1);
decisionTreeTestErrors = zeros(length(TREE_MAX_DEPTHS), 1);

cnt = 1;
for depth = TREE_MAX_DEPTHS
    fprintf('Decision tree max depth: %d\n', depth);
    
    dist = 1/nTraining*ones(nTraining, 1);
    h = learnDecisionTree(XTrain, yTrain, dist, depth);
    yTrainPred = inferDecisionTree(XTrain, h);
    yTestPred = inferDecisionTree(XTest, h);
    
    eTrain = sum(yTrain ~= yTrainPred)/numel(yTrain);
    aTrain = 1-eTrain;
    fprintf('\tDecision tree training error: %f\n', eTrain);
    
    eTest = sum(yTest ~= yTestPred)/numel(yTest);
    aTest = 1-eTest;
    fprintf('\tDecision tree test error: %f\n', eTest);
    
    decisionTreeResults{cnt}.maxDepth = depth;
    decisionTreeResults{cnt}.trainAccuracy = aTrain;
    decisionTreeResults{cnt}.trainConfusionMatrix = confusionmatrix(yTrain, yTrainPred);
    decisionTreeResults{cnt}.testAccuracy = aTest;
    decisionTreeResults{cnt}.testConfusionMatrix = confusionmatrix(yTest, yTestPred);
    
    decisionTreeTrainErrors(cnt) = eTrain;
    decisionTreeTestErrors(cnt) = eTest;
    
    cnt = cnt+1;
end

%% plot evaluations
% training error for adaboost
plot(ENSEMBLE_SIZES', adaBoostTrainErrors);
xlabel('Ensemble Size');
ylabel('Training Error');
title('AdaBoost: Training Error vs. Ensemble Size');
pause;

% test error for adaboost
plot(ENSEMBLE_SIZES', adaBoostTestErrors);
xlabel('Ensemble Size');
ylabel('Test Error');
title('AdaBoost: Test Error vs. Ensemble Size');
pause;

% training error for decision tree
plot(TREE_MAX_DEPTHS', decisionTreeTrainErrors);
xlabel('Max Depth');
ylabel('Training Error');
title('Decision Tree: Training Error vs. Ensemble Size');
pause;

% test error for decision tree
plot(TREE_MAX_DEPTHS', decisionTreeTestErrors);
xlabel('Max Depth');
ylabel('Test Error');
title('Decision Tree: Test Error vs. Ensemble Size');
pause;

%% (cleanup)
rng('default') % reset seed
clearvars -except *Results