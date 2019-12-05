% compile the libsvmread.cpp
%make;

% set model parameters
lambda_U = 1e-7; lambda_V = 1e-7; d = 4;
tr = 'ratings.dat.tr'; va = 'ratings.dat.va';

% set training algorithm's parameters
epsilon = 1e-6;
max_iter = 5;
do_pcond = false;

% prepare training and test data sets
[y, W, H] = libsvmread(tr);
[y_test,W_test, H_test] = libsvmread(va);


n = max(size(W,2),size(W_test,2));
[i,j,s] = find(W);
W = sparse(i,j,s,size(W,1),n);
[i,j,s] = find(W_test);
W_test = sparse(i,j,s,size(W_test,1),n);

n = max(size(H,2),size(H_test,2));
[i,j,s] = find(H);
H = sparse(i,j,s,size(H,1),n);
[i,j,s] = find(H_test);
H_test = sparse(i,j,s,size(H_test,1),n);

%Init freq regularization
U_reg = sum(W)'*lambda_U;
V_reg = sum(H)'*lambda_V;

% learn an FM model
[U, V] = fm_train(y, W, H, U_reg, V_reg, d, epsilon, max_iter, do_pcond, y_test, W_test, H_test);

% do prediction
%y_tilde = fm_predict(X_test, w, U, V);
%display(sprintf('test accuracy: %f', sum(sign(y_tilde) == y_test)/size(y_test,1)));
