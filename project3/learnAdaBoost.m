function [ hypothesisStruct ] = learnAdaBoost( data, labels, nEnsembles )
%LEARNADABOOST Learn ensemble of hypotheses with AdaBoost.
%   data:               data matrix; rows = examples, cols = features
%   labels:             labels associated with data
%   nEnsembles:         number of ensembles, a.k.a. number of iterations
%   hypothesisStruct:   struct containing hypotheses and weights
%                       each hypothesis:
%                           .h index on feature to decide
%                           .alpha weight for weighted vote in the end

%% setup
[nExamples, ~] = size(data);
hypothesisStruct.h = zeros(nEnsembles, 1);
hypothesisStruct.alpha = zeros(nEnsembles, 1);

% set initial weights (distribution) to uniform
distribution = 1/nExamples*ones(nExamples, 1);

%% main loop
for l = 1:nEnsembles
    % get weak classifier hypothesis and error
    h_l = learnDecisionStump(data, labels, distribution);
    predictedLabels = inferDecisionStump(data, h_l);
    incorrectMask = double(predictedLabels ~= labels);
    e_l = sum(distribution .* incorrectMask);
    
    % stop if error >= 1/2
    if e_l >= 0.5
        break;
    end
    
    % compute weight for the weak classifier
    alpha_l = 1/2*log((1-e_l)/e_l);
    
    % update distribution for training data
    distribution = distribution .* (incorrectMask*exp(alpha_l)...
        + (1-incorrectMask)*exp(-alpha_l));
    distribution = distribution ./ sum(distribution);
    
    % keep hypothesis and alpha for final classifier
    hypothesisStruct.h(l) = h_l;
    hypothesisStruct.alpha(l) = alpha_l;
end

end

