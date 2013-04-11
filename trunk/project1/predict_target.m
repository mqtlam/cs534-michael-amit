function [ y ] = predict_target( X, w, ADD_BIAS_TERM )
%PREDICT_TARGET Predict target value of linear regression.
%   X:              MxN feature matrix
%                       M examples
%                       N features
%   w:              learned weight/parameter vector
%   y:              Mx1 matrix of target predictions
%   ADD_BIAS_TERM:  0 (default): leave X alone
%                   1: append bias terms to X

if nargin < 3
    ADD_BIAS_TERM = 0;
end

if ADD_BIAS_TERM
    nExamples = size(X, 1);
    X = [X ones(nExamples, 1)];
end

y = X*w;

end

