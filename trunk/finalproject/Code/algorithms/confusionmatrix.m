function [ matrix ] = confusionmatrix( groundtruth, predictions, raw )
%CONFUSIONMATRIX compute confusion matrix in format we want
%
%               groundtruth
%                 1 -1
%               +--+--+
% predictions  1|  |  |
%               +--+--+
%            -1 |  |  |
%               +--+--+

if nargin < 3
    raw = logical(0);
end

matrix = confusionmat(groundtruth, predictions);
matrix = rot90(matrix', 2);

if ~raw
    matrix(:, 1) = matrix(:, 1)./sum(matrix(:, 1));
    matrix(:, 2) = matrix(:, 2)./sum(matrix(:, 2));
end

end

