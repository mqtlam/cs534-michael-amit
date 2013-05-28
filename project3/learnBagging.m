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
hypothesisStruct.h = cell(nEnsembles, 1);

% set the number of samples. 'T'
T = size(data, 1);

%% main loop
for l = 1:nEnsembles
    % randomly select 'T' samples with replacement from the data
    dataSample = datasample(data, T);
    
    % keep hypothesis for final classifier
    trainLabels = dataSample(:, 1);
    trainData = dataSample(:, 2:end);
    hypothesisStruct.h{l} = learnDecisionStump(trainData, trainLabels);
end

end

