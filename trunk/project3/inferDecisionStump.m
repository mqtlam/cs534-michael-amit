function [ predictedLabels ] = inferDecisionStump( data, hypothesis )
%INFERDECISIONSTUMP Given data and hypothesis, predict labels.
%   data:               data matrix; rows = examples, cols = features
%   hypothesis:         index on feature to decide
%   predictedLabels:    predicted labels from decision stump

index = abs(hypothesis);
test = hypothesis > 0;

predictedLabels = double(data(:, index) == test);

end

