function [ hypothesis ] = learnDecisionStump( data, labels, distribution )
%LEARNDECISIONSTUMP Learn a decision stump hypothesis given data, labels
%and distribution (weights over data).
%   data:           data matrix; rows = examples, cols = features
%   labels:         labels associated with data
%   distribution:   weights over data (elements must sum to one)
%                   if not provided, distribution is uniform
%   hypothesis:     decision stump hypothesis
%                       .feature is feature to split (feature index)
%                       .neg is label of negative branch/leaf (0 or 1)
%                       .pos is label of positive branch/leaf (0 or 1)

%% setup
[nExamples, nFeatures] = size(data);

% default: use uniform distribution
if nargin < 3
    distribution = 1/nExamples*ones(nExamples, 1);
end

%% learn hypothesis
entropies = zeros(nFeatures, 1);
for feature = 1:nFeatures
    % number of weighted positive and negative examples
    pos = sum(distribution.*labels);
    neg = sum(distribution.*(1-labels));
    
    % number of weighted positive and negative examples
    % given selected feature is positive
    featpos_pos = sum(distribution.*data(:, feature).*labels);
    featpos_neg = sum(distribution.*data(:, feature).*(1-labels));
    
    % number of weighted positive and negative examples
    % given selected feature is negative
    featneg_pos = sum(distribution.*(1-data(:, feature)).*labels);
    featneg_neg = sum(distribution.*(1-data(:, feature)).*(1-labels));
    
    % compute probabilities and entropies
    P_X_pos = (featpos_pos+featpos_neg)/(pos+neg);
    
    P_X_pos_1 = featpos_pos/(featpos_pos+featpos_neg);
    H_Y_X_pos = calcEntropy([P_X_pos_1 1-P_X_pos_1]);
    
    P_X_neg_1 = featneg_pos/(featneg_pos+featneg_neg);
    H_Y_X_neg = calcEntropy([P_X_neg_1 1-P_X_neg_1]);
    
    % compute entropy for feature
    entropies(feature) = H_Y_X_pos*P_X_pos + H_Y_X_neg*(1-P_X_pos);
end

% select minimum entropy (same as selecting maximum information gain)
[~, hypothesis.feature] = min(entropies);

% recompute weighted positive and negative examples given best feature
featpos_pos = sum(distribution.*data(:, hypothesis.feature).*labels);
featpos_neg = sum(distribution.*data(:, hypothesis.feature).*(1-labels));
featneg_pos = sum(distribution.*(1-data(:, hypothesis.feature)).*labels);
featneg_neg = sum(distribution.*(1-data(:, hypothesis.feature)).*(1-labels));

% set leaf node labels to majority class
hypothesis.neg = double(featneg_pos > featneg_neg);
hypothesis.pos = double(featpos_pos > featpos_neg);

end

function entropy = calcEntropy(probs)
entropy = -sum(probs.*log2(probs));
if isnan(entropy) % one of probs is 0
    entropy = 0;
end
end

