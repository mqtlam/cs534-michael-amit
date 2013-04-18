function [ w, plot ] = batch_perceptron( X, y )
% BATCH_PERCEPTRON Batch perceptron classification.
%   X:      MxN data
%               M data entries
%               N dimensions (no bias term)
%   y:      Mx1 matrix of class labels
%   w:      learned weight/parameter vector
%   plot:   vector of number of classification errors for each step 

[nExamples, nFeatures] = size(X);

% constants
EPSILON = .0001;

% augment X with bias terms
X_aug = [ones(nExamples,1) X];

% initialize parameters
w = zeros(nFeatures + 1, 1);
w_prev = w;

% main loop
step = 1;
while (1)
    delta = zeros(nFeatures + 1, 1);
    
    for m = 1:nExamples
        u = w' * X_aug(m,:)';
        if y(m)*u <= 0
            delta = delta - y(m)' * X_aug(m,:)';
        end
    end
    delta = delta/(nFeatures+1);
    w = w - delta;
    if norm(w_prev-w) < EPSILON
        break;
    end
    plot(step) = 0;
    for i=1:nExamples
        class = w(:)' * X_aug(i,:)';
        if ((class <= 0 && y(i) >0) || (class > 0 && y(i) <= 0))
            plot(step) = plot(step) + 1;
        end
    end
    step = step + 1;
    w_prev = w;
end

end

