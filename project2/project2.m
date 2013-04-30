%% Main script

%% load data
trainData = dlmread('data/train.data', ' ');
trainLabels = dlmread('data/train.label', ' ');
testData = dlmread('data/test.data', ' ');
testLabels = dlmread('data/test.label', ' ');

fid = fopen('data/newsgrouplabels.txt');
newsgrouplabels = textscan(fid, '%s');
fclose(fid);
newsgrouplabels = newsgrouplabels{1};

fid = fopen('data/vocabulary.txt');
vocabulary = textscan(fid, '%s');
fclose(fid);
vocabulary = vocabulary{1};

%% constants
nClasses = length(newsgrouplabels);
dictSize = length(vocabulary);

%% train classifier for bernoulli model
tic;
[likelihood, prior] = learn_NB_bernoulli(trainData, trainLabels, nClasses, dictSize);
trainingTime = toc;
fprintf('Time to train for bernoulli: %f seconds\n', trainingTime);

%% test classifier for bernoulli model
tic;
predictLabels = inference_NB_bernoulli(testData, likelihood, prior);
testingTime = toc;
fprintf('Time to test for bernoulli: %f seconds\n', testingTime);

[accuracy, confusionMat] = evaluate_prediction(predictLabels, testLabels);
fprintf('Prediction accuracy for bernoulli: %f\n', accuracy);
fprintf('Confusion matrix for bernoulli:\n');
heatmap(confusionMat, 1:nClasses, 1:nClasses, 1);
pause;

%% train classifier for multinomial model
tic;
[likelihood, prior] = learn_NB_multinomial(trainData, trainLabels, nClasses, dictSize);
trainingTime = toc;
fprintf('Time to train for multinomial: %f seconds', trainingTime);

%% test classifier for multinomial model
tic;
predictLabels = inference_NB_multinomial(testData, likelihood, prior);
testingTime = toc;
fprintf('Time to test for multinomial: %f seconds', testingTime);

[accuracy, confusionMat] = evaluate_prediction(predictLabels, testLabels);
fprintf('Prediction accuracy for multinomial: %f\n', accuracy);
fprintf('Confusion matrix for multinomial:\n');
heatmap(confusionMat, 1:nClasses, 1:nClasses, 1);
pause;

%% (cleanup)
clearvars fid ans