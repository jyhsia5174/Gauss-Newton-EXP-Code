% compile the libsvmread.cpp
make;

% set model parameters
lambda_w = 0.0625; lambda_U = 0.0625; lambda_V = 0.0625; d = 4;

% set training algorithm's parameters
epsilon = 0.01; do_pcond = true; sub_rate = 0.1;

% prepare training and test data sets
[y,X] = libsvmread('fourclass_scale.tr');
[y_test,X_test] = libsvmread('fourclass_scale.te');
n = max(size(X,2),size(X_test,2));
[i,j,s] = find(X);
X = sparse(i,j,s,size(X,1),n);
[i,j,s] = find(X_test);
X_test = sparse(i,j,s,size(X_test,1),n);

% learn an FM model
[w, U, V] = fm_train(y, X, lambda_w, lambda_U, lambda_V, d, epsilon, do_pcond, sub_rate);

% do prediction
y_tilde = fm_predict(X_test, w, U, V);
display(sprintf('test accuracy: %f', sum(sign(y_tilde) == y_test)/size(y_test,1)));
