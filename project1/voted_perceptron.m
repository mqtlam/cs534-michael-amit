function [ w, plot ] = voted_perceptron( X, y )
%VOTED_PERCEPTRON Voted perceptron classification.
%   X:      MxN data
%               M data entries
%               N dimensions (no bias term)
%   y:      Mx1 matrix of class labels
%   w:      learned weight/parameter vector
%   plot:   vector of number of classification errors for each step 

[nExamples, nFeatures] = size(X);

% constants
NUM_STEPS = 100;

% augment X with bias terms
X_aug = [ones(nExamples,1) X];

% initialize parameters
w = zeros(1, nFeatures + 1);
k = 1;
c(k) = 0;
W(k,:) = w;

% main loop
step = 1;
while (step < NUM_STEPS)
    % shuffle rows
    aug = [X_aug y];
    aug = shuffle_rows(aug);
    X_aug = aug(:, 1:end-1);
    y = aug(:, end);
    
    for m = 1:nExamples
        u = W(k,:) * X_aug(m,:)';
        
        if y(m)*u <= 0
            W(k+1,:) = W(k,:) + (y(m) * X_aug(m,:));
            k = k+1;
            c(k)=0;
        else
            c(k) = c(k)+1;
        end
    end
    
    %calculating error in classification
    plot(step) = 0;
    for i=1:nExamples
        class = W(k,:) * X_aug(i,:)';
        if (class * y(i) <= 0)
            plot(step) = plot(step) + 1;
        end
    end
    step = step + 1;
end

% compute average
for i=1:k
    w_avg = c(i)*W(i,:);
end

w = w_avg;
end
