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
EPSILON = .0001;

% augment X with bias terms
X_aug = [ones(nExamples,1) X];

% initialize parameters
w = zeros(nFeatures + 1, 1);
c(1) = 0;

% main loop
step = 1;
k=1;
W(1,:) = w;
while (step<30)
    
    X_aug = shuffle_rows(X_aug);
    for m = 1:nExamples
%         u=0;
%         for n = 1:k
%             u = u + c(n) * (W(n,:) * X_aug(m,:)');
%         end
        for i=1:k
            w(:) = w(:)' + W(i,:) * c(i);
        end
        u = w' * X_aug(m,:)';
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
    disp(step);
    step = step + 1;
end
% 
% for i=1:k
%     w_avg = c(i)*W(i,:)
% end

disp (W);
disp (c);
w = W(k,:);
end
