function [ predictedLabels ] = inferDecisionStump( data, hypothesis )
%INFERDECISIONSTUMP Given data and hypothesis, predict labels.
%   data:               data matrix; rows = examples, cols = features
%   hypothesis:         decision stump hypothesis
%                           .feature is feature to split (feature index)
%                           .neg is label of negative branch/leaf (0 or 1)
%                           .pos is label of positive branch/leaf (0 or 1)
%   predictedLabels:    predicted labels from decision stump

[nExamples, ~] = size(data);
index = hypothesis.feature;

if hypothesis.neg == 0 && hypothesis.pos == 1
    predictedLabels = double(data(:, index) == 1);
elseif hypothesis.neg == 1 && hypothesis.pos == 0
    predictedLabels = double(data(:, index) == 0);
elseif hypothesis.neg == 1 && hypothesis.pos == 1
    predictedLabels = ones(nExamples, 1);
elseif hypothesis.neg == 0 && hypothesis.pos == 0
    predictedLabels = zeros(nExamples, 1);
end

% index = abs(hypothesis);
% test = hypothesis > 0;
% 
% predictedLabels = double(data(:, index) == test);

end

