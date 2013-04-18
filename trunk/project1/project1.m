%% Main script

%% (setup)
rng(12345); % fix seed

%% load data
training_data = csvread('regression-train.csv');
testing_data = csvread('regression-test.csv');

X_train = training_data(:, 1:end-1);
y_train = training_data(:, end);
X_test = testing_data(:, 1:end-1);
y_test = testing_data(:, end);

%% linear regression: batch gradient descent
fprintf('\nLinear regression w/ batch gradient descent:\n');
[w, plot_errors] = linear_regression_batch(X_train, y_train);
fprintf('Learned weights:\n');
disp(w);

y_predict = predict_target(X_test, w, 1);

[error_SSE, ~] = SSE_loss(y_predict, y_test);
fprintf('SSE on test set: %f\n', error_SSE);

plot(plot_errors);
fprintf('(displaying plot of errors...)\n');
pause;
close all
pause(0.1);

%% linear regression: stochastic gradient descent
fprintf('\nLinear regression w/ stochastic gradient descent:\n');
[w, plot_errors] = linear_regression_stochastic(X_train, y_train);
fprintf('Learned weights:\n');
disp(w);

y_predict = predict_target(X_test, w, 1);

[error_SSE, ~] = SSE_loss(y_predict, y_test);
fprintf('SSE on test set: %f\n', error_SSE);

plot(plot_errors);
fprintf('(displaying plot of errors...)\n');
pause;
close all
pause(0.1);


%% load data
classification_data = csvread('twogaussian.csv');

X_features = classification_data(:, 2:end);
y_class = classification_data(:, 1);

%% perceptron
fprintf('\nBatch Perceptron:\n');
[w, plot_errors] = batch_perceptron(X_features, y_class);
fprintf('\nLearned Weights:\n');
disp(w);
plot(plot_errors);
fprintf('(displaying plot of errors...)\n');
pause;
scatter_plot(X_features, y_class, w);
fprintf('(displaying boundary plot...)\n');
pause;
close all
pause(0.1);
%% load data
classification_data = csvread('iris-twoclass.csv');

X_features = classification_data(:, 2:end);
y_class = classification_data(:, 1);


%% voted perceptron
fprintf('\nVoted Perceptron:\n');
[w, plot_errors] = voted_perceptron(X_features, y_class);
fprintf('\nLearned Weights:\n');
disp(w);
plot(plot_errors);
fprintf('(displaying plot of errors...)\n');
pause;
scatter_plot(X_features, y_class, w);
fprintf('(displaying boundary plot...)\n');
pause;
close all
pause(0.1);
%% (cleanup)
rng('default') % reset seed