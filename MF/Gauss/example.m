% compile the libsvmread.cpp
%make;

% set model parameters
%lambda_U = 0.05; lambda_V = 0.05; d = 40;
%tr = 'ml.tr'; va = 'ml.te';

% set training algorithm's parameters
%epsilon = 1e-6;
do_pcond = false;

% prepare training and test data sets
R = mf_read(tr);
R_test = mf_read(va);
m = max(size(R,1),size(R_test,1));
n = max(size(R,2),size(R_test,2));
[i,j,s] = find(R);
R = sparse(i,j,s,m,n);
[i,j,s] = find(R_test);
R_test = sparse(i,j,s,m,n);


%Init freq regularization
IR = spones(R);
U_reg = sum(IR')'*lambda_U;
V_reg = sum(IR)'*lambda_V;
IR_test = spones(R_test);

% learn an FM model
[U, V] = fm_train(R, IR, U_reg, V_reg, d, epsilon, max_iter, do_pcond, R_test, IR_test);

% do prediction
%y_tilde = fm_predict(X_test, w, U, V);
%display(sprintf('test accuracy: %f', sum(sign(y_tilde) == y_test)/size(y_test,1)));
