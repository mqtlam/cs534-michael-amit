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

%% reduce vocabulary size by word length heuristic
% To use original vocabulary (dictionary), use below:
vocabMap = 1:length(vocabulary);
% To use new vocabulary with word length heuristic, use below:
%vocabMap = vocabTransform(vocabulary, 3);

%% constants
nClasses = length(newsgrouplabels);

%% train classifier for bernoulli model
tic;
[likelihood, prior] = learn_NB_bernoulli(trainData, trainLabels, nClasses, vocabMap);
trainingTime = toc;
fprintf('Time to train for bernoulli: %f seconds\n', trainingTime);

%% test classifier for bernoulli model
tic;
predictLabels = inference_NB_bernoulli(testData, likelihood, prior, vocabMap);
testingTime = toc;
fprintf('Time to test for bernoulli: %f seconds\n', testingTime);

%% evaluate predictions
[accuracy, confusionMat] = evaluate_prediction(predictLabels, testLabels);
fprintf('Prediction accuracy for bernoulli: %f\n', accuracy);
fprintf('Confusion matrix for bernoulli:\n');
heatmap(confusionMat, 1:nClasses, 1:nClasses, 1);
pause;

% %% train classifier for multinomial model
% tic;
% [likelihood, prior] = learn_NB_multinomial(trainData, trainLabels, nClasses, vocabMap);
% trainingTime = toc;
% fprintf('Time to train for multinomial: %f seconds\n', trainingTime);
% 
% %% test classifier for multinomial model
% tic;
% predictLabels = inference_NB_multinomial(testData, likelihood, prior, vocabMap);
% testingTime = toc;
% fprintf('Time to test for multinomial: %f seconds\n', testingTime);
% 
% %% evaluate predictions
% [accuracy, confusionMat] = evaluate_prediction(predictLabels, testLabels);
% fprintf('Prediction accuracy for multinomial: %f\n', accuracy);
% fprintf('Confusion matrix for multinomial:\n');
% heatmap(confusionMat, 1:nClasses, 1:nClasses, 1);
% pause;

%% vocabulary size study: evaluate a range of word length thresholds
THRESHOLD_RANGE = 1:5;
vocabStudyAccuracies = zeros(5, 1);
fprintf('Study of vocabulary size by allowing certain word length:\n');
i = 1;
for threshold = THRESHOLD_RANGE
    vocabMap = vocabTransform(vocabulary, threshold);

    %% train, test and evaluate on threshold
    [likelihood, prior] = learn_NB_bernoulli(trainData, trainLabels, nClasses, vocabMap);
    predictLabels = inference_NB_bernoulli(testData, likelihood, prior, vocabMap);
    [accuracy, ~] = evaluate_prediction(predictLabels, testLabels);
    
    fprintf('\nKeep words > %d letters: accuracy = %f\n', threshold, accuracy);
    vocabStudyAccuracies(i) = accuracy;
    i = i+1;
end
plot(THRESHOLD_RANGE, vocabStudyAccuracies);

%% (cleanup)
clearvars fid ans i THRESHOLD_RANGE