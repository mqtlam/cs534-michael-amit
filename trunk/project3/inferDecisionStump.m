function [ predictedLabels ] = inferDecisionStump( data, hypothesis )
%INFERDECISIONSTUMP Given data and hypothesis, predict labels.
%   data:               data matrix; rows = examples, cols = features
%   hypothesis:         signed index on feature to decide
%                       positive sign = 0->0, 1->1
%                       negative sign = 0->1, 0->1
%   predictedLabels:    predicted labels from decision stump

index = abs(hypothesis);
test = hypothesis > 0;

predictedLabels = double(data(:, index) == test);

end

