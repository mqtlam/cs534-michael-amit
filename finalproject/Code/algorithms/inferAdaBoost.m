function [ predictedLabels ] = inferAdaBoost( data, hypothesisStruct )
%INFERADABOOST Classify with AdaBoost hypotheses and weights.
%   data:               data matrix; rows = examples, cols = features
%   hypothesisStruct:   struct containing hypotheses and weights
%                       each hypothesis:
%                           .h is decision stump hypothesis
%                           .alpha is weight for weighted vote in the end
%   predictedLabels:    final predicted labels

%% setup
[nExamples, ~] = size(data);
nEnsembles = size(hypothesisStruct.h, 1);

%% final classifier
% returns linear combination of weak classifiers and weights
runningSum = zeros(nExamples, 1);
for i = 1:nEnsembles
    if hypothesisStruct.alpha(i) ~= 0
        labels = inferDecisionStump(data, hypothesisStruct.h{i});
        labels(labels == 0) = -1;
        runningSum = runningSum + hypothesisStruct.alpha(i)*labels;
    end
end

predictedLabels = double(runningSum > 0);

end

