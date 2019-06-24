function y_tilde = fm_predict(X, w, U, V)
% Predict the input instances.
% function y_tilde = fm_predict(X, w, U, V)
% Inputs:
%   X: training instances. X is an l-by-n matrix if you have l training instances in a n-dimensional feature space.
%   w: linear coefficients. An n-dimensional vector.
%   U, V: the interaction (d-by-n) matrices.
% Output:
%   y_tilde: prediction values of the input instances, an l-dimensional column vector.
    y_tilde = X*w'+0.5*(sum((U*X').*(V*X'),1))';
end
