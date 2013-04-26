function [ accuracy, confusionMat ] = evaluate_prediction( predictLabels, testLabels )
%EVALUATE_PREDICTION Evaluates the accuracy and confusion matrix for the
%predictions and ground truth labels.
%   predictLabels:      labels predicted by classifier
%   testLabels:         ground truth labels for test data
%   accuracy:           accuracy of prediction
%   confusionMat:       confusion matrix (i,j) with
%                           ground truth i, prediction j

accuracy = sum(predictLabels == testLabels)/length(testLabels);
confusionMat = confusionmat(testLabels, predictLabels);

end
