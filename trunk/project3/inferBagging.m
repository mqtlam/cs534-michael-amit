function [ predictedLabels ] = inferBagging( data, hypothesisStruct )
%INFERBAGGING Classify with Bagging hypothesis and majority vote.
%   data:               data matrix; rows = examples, cols = features
%   hypothesisStruct:   struct containing hypotheses
%                       each hypothesis:
%                           .h is decision stump hypothesis
%   predictedLabels:    final predicted labels

%% setup
[nExamples, ~] = size(data);
nEnsembles = size(hypothesisStruct.h, 1);

%% final classifier: returns majority vote
positives = zeros(nExamples, 1);
negatives = zeros(nExamples, 1);
for i = 1:nEnsembles
    h = inferDecisionStump(data, hypothesisStruct.h{i});
    positives = positives + double(h == 1);
    negatives = negatives + double(h == 0);
end

predictedLabels = double(positives > negatives);

end

