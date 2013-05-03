function [ predictLabels ] = inference_NB_multinomial( testData, likelihood, prior, vocabMap )

%INFERENCE_NB_BERNOULLI Perform inference with trained Naive Bayes 
%   classifier with Bernoulli model.
%
%   testData:       test data, [docId wordId count] columns
%   likelihood:     likelihood probabilities, VxK matrix
%                       V vocabulary size, K classes
%   prior:          prior probabilities, K vector
%                       K classes
%   vocabMap:       mapping from original vocabulary to new vocabulary
%   predictLabels:  class predictions for each document

%% initialization
[dictSize, nClasses] = size(likelihood);
nDocs = testData(end, 1); % assume data sorted; original test data file!

% class label predictions for each document
predictLabels = zeros(nDocs, 1);

%% classify each document
index = 1;
for doc = 1:nDocs
    %% compute bag of words feature for this document
    % assume data sorted; original test data file!
    bag = zeros(dictSize, 1);
    while (index <= size(testData, 1) && testData(index, 1) == doc)
        wordId = testData(index, 2);
        wordId = vocabMap(wordId); % vocabulary mapping
        count = testData(index,3);
        if wordId == 0 % vocabulary mapping
            index = index + 1;
            continue;
        end
        
        bag(wordId) = count;
        index = index + 1;
    end
    
    %% calculate class label probabilities using Bayes Rule
    % goal is to find maximum the posterior probability
    % so classProbs doesn't include p(x), the constant normalization factor
    % note: operating on log of probabilities
    classProbs = zeros(nClasses, 1);
    for class = 1:nClasses
        classProbs(class) = sum( bag.*likelihood(:, class))...
            + prior(class);
    end
    
    %% perform inference using Decision Theory
    % decision rule is to select the class that maximizes the posterior
    % note: max of log of probabilities = max of probabilities
    [~, predictLabels(doc)] = max(classProbs);
end

end

