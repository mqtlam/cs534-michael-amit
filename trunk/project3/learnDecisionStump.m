function [ hypothesis ] = learnDecisionStump( data, labels, weights )
%LEARNDECISIONSTUMP Learn a decision stump hypothesis given data, labels
%and weights (distribution over data).
%   data:           data matrix; rows = examples, cols = features
%   labels:         labels associated with data
%   weights:        weights over data (elements must sum to one)
%                   if not provided, weights not used
%   hypothesis:     index on feature to decide

%% setup
[nExamples, nFeatures] = size(data);

% default: don't use weights by setting all individual weights to 1
if nargin < 3
    weights = ones(nExamples, 1);
end

%% learn hypothesis
entropies = zeros(nFeatures, 1);
for feature = 1:nFeatures
    % calculate probabilities
    P_X_pos = sum(weights.*double(data(:, feature) == 1))/sum(weights);
    P_X_neg = 1-P_X_pos;
    
    % calculate entropy given feature is positive
    P_X_pos_1 = sum(weights.*data(:, feature).*labels)...
        / sum(weights.*double(data(:, feature) == 1));
    H_Y_X_pos = calcEntropy([P_X_pos_1 1-P_X_pos_1]);
    
    % calculate entropy given feature is negative
    P_X_neg_1 = sum(weights.*(1-data(:, feature)).*labels)...
        / sum(weights.*double(data(:, feature) == 0));
    H_Y_X_neg = calcEntropy([P_X_neg_1 1-P_X_neg_1]);
    
    entropies(feature) = H_Y_X_pos*P_X_pos + H_Y_X_neg*P_X_neg;
end

% select minimum entropy (same as maximum information gain)
[~, hypothesis] = min(entropies);

% pick best test direction
predictedLabels = inferDecisionStump(data, hypothesis);
errorPos = sum(predictedLabels ~= labels);
errorNeg = sum((1-predictedLabels) ~= labels);
if errorPos > errorNeg
    hypothesis = -hypothesis;
end

end

function entropy = calcEntropy(probs)
entropy = -sum(probs.*log2(probs));
if isnan(entropy) % one of probs is 0
    entropy = 0;
end
end

