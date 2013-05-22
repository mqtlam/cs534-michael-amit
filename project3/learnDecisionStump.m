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
    P_X_pos = sum(weights.*double(labels == 1))/sum(weights);
    P_X_neg = sum(weights.*double(labels == 0))/sum(weights);
    
    pos = data(:, feature).*labels;
    neg = data(:, feature).*(1-labels);
    
    % calculate entropy given feature is positive
    P_X_pos_1 = sum(weights.*double(pos == 1))...
        / sum(weights.*double(labels == 1));
    P_X_pos_0 = 1-P_X_pos_1;
    H_Y_X_pos = calcEntropy([P_X_pos_1 P_X_pos_0]);
    
    % calculate entropy given feature is negative
    P_X_neg_1 = sum(weights.*double(neg == 1))...
        / sum(weights.*double(labels == 0));
    P_X_neg_0 = 1-P_X_neg_1;
    H_Y_X_neg = calcEntropy([P_X_neg_1 P_X_neg_0]);
    
    entropies(feature) = H_Y_X_pos*P_X_pos + H_Y_X_neg*P_X_neg;
end

[~, hypothesis] = min(entropies);

end

function entropy = calcEntropy(probs)
entropy = -sum(probs.*log2(probs));
end

