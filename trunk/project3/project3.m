%% Main script

%% load data
train = importdata('data/SPECT-train.csv');
trainLabels = train.data(:, 1);
trainData = train.data(:, 2:end);

test = importdata('data/SPECT-test.csv');
testLabels = test.data(:, 1);
testData = test.data(:, 2:end);

clearvars train test

%% constants
ensembleSizes = [5, 10, 15, 20, 25, 30];

%% bagging
%TODO

%% AdaBoost
trainingErrors = zeros(length(ensembleSizes), 1);
testingErrors = zeros(length(ensembleSizes), 1);

% experiment over different ensemble sizes
for i = 1:length(ensembleSizes)
    size = ensembleSizes(i);
    
    % learn ensemble hypothesis
    hypothesis = learnAdaBoost(trainData, trainLabels, size);
    
    % test on training examples
    predictedTrainingLabels = inferAdaBoost(trainData , hypothesis);
    trainingErrors(i) = sum(predictedTrainingLabels ~= trainLabels);
    
    % test on test examples
    predictedTestingLabels = inferAdaBoost(testData , hypothesis);
    testingErrors(i) = sum(predictedTestingLabels ~= testLabels);
end

% plot training and testing errors
plot(ensembleSizes, trainingErrors, 'ks--');
pause;
plot(ensembleSizes, testingErrors, 'ks--');
pause;
