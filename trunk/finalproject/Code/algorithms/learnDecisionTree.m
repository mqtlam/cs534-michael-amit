function [ hypothesis ] = learnDecisionTree( data, labels, distribution, maxDepth, hypothesis, currentNode, currentDepth )
%LEARNDECISIONTREE Learn a decision tree hypothesis given data, labels
%and distribution (weights over data).
%   data:               data matrix; rows = examples, cols = features
%   labels:             labels associated with data
%   distribution:       weights over data (elements must sum to one)
%                       if not provided, distribution is uniform
%   maxDepth:           maximum tree depth (default 10)
%   hypothesis:         decision tree hypothesis (for recursion)
%                           each node:
%                           .feature is feature to split (feature index)
%                           .threshold is threshold to test on feature
%                           .neg is child node or leaf label of negative branch
%                           .pos is child node or leaf label of positive branch

%% setup
DEFAULT_MAX_DEPTH = 10;

[nExamples, nFeatures] = size(data);

% default: use uniform distribution
if nargin < 3
    distribution = 1/nExamples*ones(nExamples, 1);
end

% normalize distribution if it hasn't been already
distribution = distribution./sum(distribution);

if nargin < 4
    maxDepth = DEFAULT_MAX_DEPTH;
end

% initialize root node of decision tree
if nargin < 5
    currentNode = 1;
    currentDepth = 0;
end

hypothesis{currentNode}.feature = 0;
hypothesis{currentNode}.threshold = 0;
hypothesis{currentNode}.neg = 0;
hypothesis{currentNode}.pos = 0;

%% learn hypothesis
entropies = zeros(nFeatures, 1);
bestThresholds = zeros(nFeatures, 1);
for feature = 1:nFeatures
    % compute feature threshold: sort examples
    augData = [data(:, feature), labels, distribution];
    augData = sortrows(augData, 1);
    newData = augData(:, 1);
    newLabels = augData(:, 2);
    newDist = augData(:, 3);
    
    % slide threshold, keeping track of entropy
    thresholds = []; % [threshold, entropy]
    prevLabel = -.314; % arbitrary value not in label space
    for i = 1:length(labels)
        if prevLabel == labels(i) % only test entropy at label changes
            continue;
        end
        
        threshold = newData(i);
        newDataFeature = thresholdData(newData, threshold);
        entropy = ...
            calcFeatureEntropy(newDataFeature, newLabels, newDist);
        thresholds = [thresholds; [threshold, entropy]];
        
        prevLabel = labels(i);
    end
    
    % pick best threshold and apply threshold to feature
    [~, ind] = min(thresholds(:, 2));
    bestThresholds(feature) = thresholds(ind, 1);
    newDataFeature = thresholdData(newData, bestThresholds(feature));
    
    % compute entropy for feature
    entropies(feature) = ...
        calcFeatureEntropy(newDataFeature, newLabels, newDist);
end

% select minimum entropy (same as selecting maximum information gain)
[~, hypothesis{currentNode}.feature] = min(entropies);
hypothesis{currentNode}.threshold = bestThresholds(hypothesis{currentNode}.feature);

if currentDepth + 1 > maxDepth
    
    % apply threshold to best feature for next operations...
    dataFeature = thresholdData(data(:, hypothesis{currentNode}.feature), hypothesis{currentNode}.threshold);
    
    % recompute weighted positive and negative examples given best feature
    featpos_pos = sum(distribution.*dataFeature.*labels);
    featpos_neg = sum(distribution.*dataFeature.*(1-labels));
    featneg_pos = sum(distribution.*(1-dataFeature).*labels);
    featneg_neg = sum(distribution.*(1-dataFeature).*(1-labels));
    
    % set leaf node labels to majority class
    hypothesis{currentNode}.neg = -1*double(featneg_pos > featneg_neg);
    hypothesis{currentNode}.pos = -1*double(featpos_pos > featpos_neg);
    
else
    
    % apply threshold to best feature for next operations...
    dataFeature = thresholdData(data(:, hypothesis{currentNode}.feature), hypothesis{currentNode}.threshold);
    
    % split data based on threshold
    augData = [data labels distribution];
    dataPos = augData(logical(dataFeature), :);
    dataNeg = augData(~logical(dataFeature), :);
    
    % recursively generate tree on positive child branch
    if size(dataPos, 1) > 0
        
        data = dataPos(:, 1:end-2);
        labels = dataPos(:, end-1);
        distribution = dataPos(:, end);
        hypothesis{currentNode}.pos = length(hypothesis) + 1;
        hypothesis = learnDecisionTree(data, labels, distribution, maxDepth, hypothesis, length(hypothesis) + 1, currentDepth+1);
    
    else
        
        % apply threshold to best feature for next operations...
        dataFeature = thresholdData(data(:, hypothesis{currentNode}.feature), hypothesis{currentNode}.threshold);

        % recompute weighted positive and negative examples given best feature
        featpos_pos = sum(distribution.*dataFeature.*labels);
        featpos_neg = sum(distribution.*dataFeature.*(1-labels));
        featneg_pos = sum(distribution.*(1-dataFeature).*labels);
        featneg_neg = sum(distribution.*(1-dataFeature).*(1-labels));

        % set leaf node labels to majority class
        hypothesis{currentNode}.neg = -1*double(featneg_pos > featneg_neg);
        hypothesis{currentNode}.pos = -1*double(featpos_pos > featpos_neg);
        
    end
    
    % recursively generate tree on negative child branch
    if size(dataNeg, 1) > 0
        
        data = dataNeg(:, 1:end-2);
        labels = dataNeg(:, end-1);
        distribution = dataNeg(:, end);
        hypothesis{currentNode}.neg = length(hypothesis) + 1;
        hypothesis = learnDecisionTree(data, labels, distribution, maxDepth, hypothesis, length(hypothesis) + 1, currentDepth+1);
    
    else
       
        % apply threshold to best feature for next operations...
        dataFeature = thresholdData(data(:, hypothesis{currentNode}.feature), hypothesis{currentNode}.threshold);

        % recompute weighted positive and negative examples given best feature
        featpos_pos = sum(distribution.*dataFeature.*labels);
        featpos_neg = sum(distribution.*dataFeature.*(1-labels));
        featneg_pos = sum(distribution.*(1-dataFeature).*labels);
        featneg_neg = sum(distribution.*(1-dataFeature).*(1-labels));

        % set leaf node labels to majority class
        hypothesis{currentNode}.neg = -1*double(featneg_pos > featneg_neg);
        hypothesis{currentNode}.pos = -1*double(featpos_pos > featpos_neg);
        
    end
end

end

function entropy = calcFeatureEntropy(featureData, labels, distribution)
    % number of weighted positive and negative examples
    pos = sum(distribution.*labels);
    neg = sum(distribution.*(1-labels));
    
    % number of weighted positive and negative examples
    % given selected feature is positive
    featpos_pos = sum(distribution.*featureData.*labels);
    featpos_neg = sum(distribution.*featureData.*(1-labels));
    
    % number of weighted positive and negative examples
    % given selected feature is negative
    featneg_pos = sum(distribution.*(1-featureData).*labels);
    featneg_neg = sum(distribution.*(1-featureData).*(1-labels));
    
    % compute probabilities and entropies
    P_X_pos = (featpos_pos+featpos_neg)/(pos+neg);
    
    P_X_pos_1 = featpos_pos/(featpos_pos+featpos_neg);
    H_Y_X_pos = calcEntropy([P_X_pos_1 1-P_X_pos_1]);
    
    P_X_neg_1 = featneg_pos/(featneg_pos+featneg_neg);
    H_Y_X_neg = calcEntropy([P_X_neg_1 1-P_X_neg_1]);
    
    % return entropy
    entropy = H_Y_X_pos*P_X_pos + H_Y_X_neg*(1-P_X_pos);
end

function entropy = calcEntropy(probs)
    entropy = -sum(probs.*log2(probs));
    if isnan(entropy) % one of probs is 0
        entropy = 0;
    end
end

function threshData = thresholdData(data, threshold)
    threshData = double(data > threshold);
end