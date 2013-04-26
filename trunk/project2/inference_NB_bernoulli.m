function [ predictLabels ] = inference_NB_bernoulli( testData, likelihood, prior )
%INFERENCE_NB_BERNOULLI Perform inference with trained Naive Bayes 
%   classifier with Bernoulli model.
%
%   testData:       test data [docId wordId count]
%   likelihood:     likelihood probabilities, VxKx2 matrix
%                       V vocabulary size, K classes, 2 for x_i=1 or 0
%   prior:          prior probabilities, K vector
%                       K classes
%   predictLabels:  class predictions for each document

%% initialization
[dictSize, nClasses, ~] = size(likelihood);
nDocs = testData(end, 1); % assume data sorted; original data file!

% class label predictions for each document
predictLabels = zeros(nDocs, 1);

%% classify each document
for doc = 1:nDocs
    %% compute bag of words feature for this document
    bag = zeros(dictSize, 1);
    testDataForDoc = testData(testData(:,1) == doc, 2);
    bag(testDataForDoc) = 1;
    
    %% calculate class label probabilities using Bayes Rule
    % goal is to find maximum the posterior probability
    % so classProbs doesn't include p(x), the constant normalization factor
    classProbs = zeros(nClasses, 1);
    for class = 1:nClasses
        classProbs(class) = sum( bag.*likelihood(:, class, 1)...
            + (1-bag).*likelihood(:, class, 2) )...
            + prior(class);
    end
    
    %% perform inference using Decision Theory
    % decision rule is to select the class that maximizes the posterior
    % max of log of probabilities = max of probabilities
    [~, predictLabels(doc)] = max(classProbs);
end

end

