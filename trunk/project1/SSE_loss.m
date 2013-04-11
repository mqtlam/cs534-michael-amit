function [ error, grad ] = SSE_loss( y_predict, y, X )
%SSE_LOSS Compute sum of squared error and gradient.
%   y_predict:  Mx1 matrix of target predictions
%   y:          Mx1 matrix of ground truth predictions
%   error:      SSE computed for each y
%   grad:       SSE gradient computed for each y
%   X:          (optional) MxN feature matrix for computing gradient
%                   M examples
%                   N features

% compute sum of squared error
error = sum((y_predict - y).^2);

% compute gradient of sum of squared error
if nargin > 2
    grad = sum(X'*(y_predict - y), 2);
else
    grad = zeros(size(y));
end

end

