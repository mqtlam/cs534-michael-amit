clear;

%% loop over datasets
%for iter = [2,4]
    %fprintf('\nUSING DATASET %d...\n\n', iter);

    %% settings
    ENSEMBLE_SIZES = [5, 10, 15, 20, 25, 30, 35, 40];
    TREE_MAX_DEPTHS = [0, 1, 2, 3, 4, 5, 6, 7];

    PERCENT_TRAINING = .75; % percentage of dataset allocated for training
    RANDOMIZE_DATASET = 1;  % randomize dataset for training/test split
    FIX_SEED = 1;           % fix random seed or not

    if FIX_SEED
        rng(12345); % fix seed
    end

    PAUSE_PLOT = 0;
    
    %SAVE_FILENAME = 'dataset2';
    %SAVE_FILENAME = 'dataset4';
    %SAVE_FILENAME = sprintf('dataset%d', iter);
    SAVE_FILENAME = 'dataset';
    
    %% load data
    %load('../ProjectDataset/game2/dataset2.mat');
    %load('../ProjectDataset/game4/dataset4.mat');
    %load(sprintf('../ProjectDataset/game%d/dataset%d.mat', iter, iter));
    
    X = [];
    y = [];
    for game = [2,3,4]
        load(sprintf('../ProjectDataset/game%d/dataset%d.mat', game, game));
        %load(sprintf('../ProjectDataset/game%d/vidnames%d.mat', game, game));

        % features
        %X = pData(:, 1:end-1);
        X = vertcat(X, pData(:, 1:end-1));
        [nExamples, nFeatures] = size(X);

        % ground truth
        %y = pData(:, end);
        y = vertcat(y, pData(:, end));
        y(y == 2) = 0;
    end

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
        adaBoostResults{cnt}.h = h;
        adaBoostResults{cnt}.trainAccuracy = aTrain;
        adaBoostResults{cnt}.trainRawConfusionMatrix = confusionmatrix(yTrain, yTrainPred, 1);
        adaBoostResults{cnt}.trainConfusionMatrix = confusionmatrix(yTrain, yTrainPred);
        adaBoostResults{cnt}.testAccuracy = aTest;
        adaBoostResults{cnt}.testRawConfusionMatrix = confusionmatrix(yTest, yTestPred, 1);
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
        decisionTreeResults{cnt}.h = h;
        decisionTreeResults{cnt}.trainAccuracy = aTrain;
        decisionTreeResults{cnt}.trainRawConfusionMatrix = confusionmatrix(yTrain, yTrainPred, 1);
        decisionTreeResults{cnt}.trainConfusionMatrix = confusionmatrix(yTrain, yTrainPred);
        decisionTreeResults{cnt}.testAccuracy = aTest;
        decisionTreeResults{cnt}.testRawConfusionMatrix = confusionmatrix(yTest, yTestPred, 1);
        decisionTreeResults{cnt}.testConfusionMatrix = confusionmatrix(yTest, yTestPred);

        decisionTreeTrainErrors(cnt) = eTrain;
        decisionTreeTestErrors(cnt) = eTest;

        cnt = cnt+1;
    end

    %% plot evaluations
    % training error for adaboost
    plot(ENSEMBLE_SIZES', adaBoostTrainErrors, 's--');
    xlabel('Ensemble Size');
    ylabel('Training Error');
    title('AdaBoost: Training Error vs. Ensemble Size');
    print('-djpeg', strcat(SAVE_FILENAME, '_boost_train.jpg'));
    if PAUSE_PLOT
        pause;
    end

    % test error for adaboost
    plot(ENSEMBLE_SIZES', adaBoostTestErrors, 's--');
    xlabel('Ensemble Size');
    ylabel('Test Error');
    title('AdaBoost: Test Error vs. Ensemble Size');
    print('-djpeg', strcat(SAVE_FILENAME, '_boost_test.jpg'));
    if PAUSE_PLOT
        pause;
    end

    % training error for decision tree
    plot(TREE_MAX_DEPTHS', decisionTreeTrainErrors, 's--');
    xlabel('Max Depth');
    ylabel('Training Error');
    title('Decision Tree: Training Error vs. Ensemble Size');
    print('-djpeg', strcat(SAVE_FILENAME, '_tree_train.jpg'));
    if PAUSE_PLOT
        pause;
    end

    % test error for decision tree
    plot(TREE_MAX_DEPTHS', decisionTreeTestErrors, 's--');
    xlabel('Max Depth');
    ylabel('Test Error');
    title('Decision Tree: Test Error vs. Ensemble Size');
    print('-djpeg', strcat(SAVE_FILENAME, '_tree_test.jpg'));
    if PAUSE_PLOT
        pause;
    end
    
    close all;
    pause(1);

    % save results
    save(strcat(SAVE_FILENAME, '_results.mat'), 'adaBoostResults', 'decisionTreeResults');

    %% (cleanup)
    rng('default') % reset seed
    clearvars -except iter

%end

clear;