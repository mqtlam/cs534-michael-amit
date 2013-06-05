function [ predictedLabels ] = inferDecisionTree( data, hypothesis )
%INFERDECISIONTREE Given data and hypothesis, predict labels.
%   data:               data matrix; rows = examples, cols = features
%   hypothesis:         decision tree hypothesis
%                           .feature is feature to split (feature index)
%                           .threshold is threshold to test on feature
%                           .neg is child node or leaf label of negative branch
%                           .pos is child node or leaf label of positive branch
%   predictedLabels:    predicted labels from decision stump

[nExamples, ~] = size(data);

predictedLabels = -1*ones(nExamples, 1);

for i = 1:nExamples
    currentNode = 1; % root node
    while currentNode > 0
        feature = hypothesis{currentNode}.feature;
        thresh = hypothesis{currentNode}.threshold;
        
        branch = thresholdData(data(i, feature), thresh);
        %branch = double(dataThresh == 1);
        
        if branch == 1
            if hypothesis{currentNode}.pos == -1
                predictedLabels(i) = 1;
                currentNode = 0;
            elseif hypothesis{currentNode}.pos == 0
                predictedLabels(i) = 0;
                currentNode = 0;
            else
                currentNode = hypothesis{currentNode}.pos;
            end
        elseif branch == 0
            if hypothesis{currentNode}.neg == -1
                predictedLabels(i) = 1;
                currentNode = 0;
            elseif hypothesis{currentNode}.neg == 0
                predictedLabels(i) = 0;
                currentNode = 0;
            else
                currentNode = hypothesis{currentNode}.neg;
            end
        end
    end
end

end

function threshData = thresholdData(data, threshold)
    threshData = double(data > threshold);
end