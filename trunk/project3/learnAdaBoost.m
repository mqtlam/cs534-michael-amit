function [ hypothesisStruct ] = learnAdaBoost( data, labels, nIterations )
%LEARNADABOOST Learn ensemble of hypotheses with AdaBoost.
%   data:               data matrix; rows = examples, cols = features
%   labels:             labels associated with data
%   nIterations:        number of iterations, a.k.a. ensemble size
%   hypothesisStruct:   struct containing hypotheses and weights
%                       each hypothesis:
%                           signed index on feature to decide
%                               positive sign = 0->0, 1->1
%                               negative sign = 0->1, 0->1
%                           weight (alpha) for weighted vote in the end

%% setup
[nExamples, ~] = size(data);

if nargin < 3
    nIterations = 10;
end

hypothesisStruct.h = zeros(nIterations, 1);
hypothesisStruct.alpha = zeros(nIterations, 1);
    
% set initial weights (distribution) to uniform
distribution = 1/nExamples*ones(nExamples, 1);

%% main loop
for l = 1:nIterations
    % get weak classifier hypothesis and error
    h_l = learnDecisionStump(data, labels, distribution);
    predictLabels = inferDecisionStump(data, h_l);
    e_l = sum(distribution .* (predictLabels ~= labels));
    
    % stop if error > 1/2
    if e_l >= 0.5
        break;
    end
    
    % compute weight for the weak classifier
    alpha_l = 1/2*log((1-e_l)/e_l);
    
    % update weights (distribution) for training data
    predictedLabels = inferDecisionStump(data, h_l);
    incorrectMask = double(predictedLabels == labels);
    distribution = distribution*(incorrectMask*exp(alpha_l)...
        + (1-incorrectMask)*exp(-alpha_l));
    distribution = distribution./sum(distribution);
    
    % keep records
    hypothesisStruct.h(l) = h_l;
    hypothesisStruct.alpha(l) = alpha_l;
end

end

