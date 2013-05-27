function [ predictedLabels ] = inferBagging( data, hypothesisStruct )
%INFERBAGGING Classify with Bagging hypothesis and majority vote.
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
voting = zeros(size(data,2),1);
for i = 1:nEnsembles
    voting(hypothesisStruct.h{i}.feature) = voting(hypothesisStruct.h{i}.feature) + 1; 
end

[~, index] = max(voting);

for i = 1:nEnsembles
    if(hypothesisStruct.h{i}.feature == index)
        h = hypothesisStruct.h{i};
    end
end

predictedLabels = inferDecisionStump(data, h);

end

