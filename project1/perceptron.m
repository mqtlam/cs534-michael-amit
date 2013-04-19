%% load data
%rng(12345);

classification_data = csvread('iris-twoclass.csv');

X_features = classification_data(:, 2:end);
y_class = classification_data(:, 1);


%% voted perceptron
fprintf('\nVoted Perceptron:\n');
[w, plot_errors] = voted_perceptron(X_features, y_class);
fprintf('\nLearned Weights:\n');
disp(w);
plot(plot_errors);
pause;
scatter_plot(X_features, y_class, w);

rng('default') % reset seed