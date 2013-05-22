function [ predictedLabels ] = inferAdaBoost( data, hypothesisStruct )
%INFERADABOOST Classify with AdaBoost hypotheses and weights.
%   data:               data matrix; rows = examples, cols = features
%   hypothesisStruct:   struct containing hypotheses and weights
%                       each hypothesis:
%                           signed index on feature to decide
%                               positive sign = 0->0, 1->1
%                               negative sign = 0->1, 0->1
%                           weight (alpha) for weighted vote in the end
%   predictedLabels:    final predicted labels

%% setup
[nExamples, ~] = size(data);
nIterations = size(hypothesisStruct.h, 1);

%% final classifier
% returns linear combination of weak classifiers and weights
runningSum = zeros(nExamples, 1);
for i = 1:nIterations
    if hypothesisStruct.alpha(i) ~= 0
        labels = inferDecisionStump(data, hypothesisStruct.h(i));
        labels(labels == 0) = -1;
        runningSum = runningSum + hypothesisStruct.alpha(i)*labels;
    end
end

predictedLabels = sign(runningSum);
predictedLabels(predictedLabels == -1) = 0;

end

