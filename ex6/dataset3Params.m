function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

list_C = [0.01 0.03 0.1 0.3 1 3 10 30];
list_sigma = [0.01 0.03 0.1 0.3 1 3 10 30];
error = 1000;

for i = 1:length(list_C)
    tmp_c = list_C(i);
    for s = 1:length(list_sigma)
      tmp_sigma = list_sigma(s);
      model = svmTrain(X, y, tmp_c, @(x1,x2) gaussianKernel(x1,x2,tmp_sigma));
      pred = svmPredict(model, Xval);
      tmp_error = mean(double(pred ~= yval));
      if tmp_error < error,
        fprintf('\nPrediction error is %f with C %f and sigma %f', tmp_error, tmp_c, tmp_sigma);
        error = tmp_error;
        C = tmp_c;
        sigma = tmp_sigma;
      endif
    endfor
endfor


% =========================================================================

end
