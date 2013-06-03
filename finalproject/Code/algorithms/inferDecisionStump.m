function [ predictedLabels ] = inferDecisionStump( data, hypothesis )
%INFERDECISIONSTUMP Given data and hypothesis, predict labels.
%   data:               data matrix; rows = examples, cols = features
%   hypothesis:         decision stump hypothesis
%                           .feature is feature to split (feature index)
%                           .threshold is threshold to test on feature
%                           .neg is label of negative branch/leaf (0 or 1)
%                           .pos is label of positive branch/leaf (0 or 1)
%   predictedLabels:    predicted labels from decision stump

[nExamples, ~] = size(data);
index = hypothesis.feature;
thresh = hypothesis.threshold;

dataThresh = thresholdData(data(:, index), thresh);

if hypothesis.neg == 0 && hypothesis.pos == 1
    predictedLabels = double(dataThresh == 1);
elseif hypothesis.neg == 1 && hypothesis.pos == 0
    predictedLabels = double(dataThresh == 0);
elseif hypothesis.neg == 1 && hypothesis.pos == 1
    predictedLabels = ones(nExamples, 1);
elseif hypothesis.neg == 0 && hypothesis.pos == 0
    predictedLabels = zeros(nExamples, 1);
end

end

function threshData = thresholdData(data, threshold)
    threshData = double(data > threshold);
end