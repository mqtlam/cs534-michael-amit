function [ hypothesisStruct ] = learnBagging( data, nEnsembles )
%LEARNBAGGING Learn ensemble of hypothesis with Bagging.
%   data:               data matrix; rows = examples, cols = features
%   labels:             labels associated with data
%   nEnsembles:         number of ensembles, a.k.a. number of iterations
%   hypothesisStruct:   struct containing hypotheses and weights
%                       each hypothesis:
%                           .h is decision stump hypothesis
%                           .alpha is weight for weighted vote in the end

%% setup
[nExamples, ~] = size(data);
hypothesisStruct.h = cell(nEnsembles, 1);
hypothesisStruct.alpha = zeros(nEnsembles, 1);

%Set the number of samples. 'T'
T = ceil(size(data,1)/nEnsembles);

% set initial weights (distribution) to uniform
distribution = 1/nExamples*ones(T, 1);

%% main loop
for l = 1:nEnsembles
    
    
    % Randomly select 'T' samples from the data for weak classification.
    dataSample = datasample(data, T, 'Replace', false);
    
    trainLabels = dataSample(:, 1);
    trainData = dataSample(:, 2:end);
    
    % get weak classifier hypothesis and error
    h_l = learnDecisionStump(trainData, trainLabels, distribution);
    
%     predictedLabels = inferDecisionStump(data, h_l);
%     incorrectMask = double(predictedLabels ~= labels);
%     e_l = sum(distribution .* incorrectMask);
%     
%     % stop if error >= 1/2
%     if e_l >= 0.5
%         break;
%     end
%     
%     % compute weight for the weak classifier
%     alpha_l = 1/2*log((1-e_l)/e_l);
%     
%     % update distribution for training data
%     distribution = distribution .* (incorrectMask*exp(alpha_l)...
%         + (1-incorrectMask)*exp(-alpha_l));
%     distribution = distribution ./ sum(distribution);
    
    % keep hypothesis and alpha for final classifier
    hypothesisStruct.h{l} = h_l;
end

end

