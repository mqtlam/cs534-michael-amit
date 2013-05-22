function [ hypothesis ] = learnDecisionStump( data, labels, weights )
%LEARNDECISIONSTUMP Learn a decision stump hypothesis given data, labels
%and weights (distribution over data).
%   data:           data matrix; rows = examples, cols = features
%   labels:         labels associated with data
%   weights:        weights over data (elements must sum to one)
%                   if not provided, weights not used
%   hypothesis:     signed index on feature to decide
%                   positive sign = 0->0, 1->1
%                   negative sign = 0->1, 0->1

%% setup
[nExamples, nFeatures] = size(data);

% default: don't use weights by setting all individual weights to 1
if nargin < 3
    weights = ones(nExamples, 1);
end

%% learn hypothesis
infos = zeros(nFeatures, 1);
for feature = 1:nFeatures
    % TODO
    
    H_Y_X_pos = 0;
    H_Y_X_neg = 0;

    H_Y_X = H_Y_X_pos*P_X_pos + H_Y_X_neg*P_X_neg;

    I_Y_X = H_Y - H_Y_X;
    infos(feature) = I_Y_X;
end

[~, hypothesis] = min(infos);

end

function entropy = calcEntropy(probs)
entropy = -sum(probs.*log2(probs));
end
