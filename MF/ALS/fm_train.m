function [U, V] = fm_train(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test, d)
% Train a factorization machine using the proposed method in the paper below.
%   Wei-Sheng Chin, Bo-Wen Yuan, Meng-Yuan Yang, and Chih-Jen Lin, An Efficient Alternating Newton Method for Learning Factorization Machines, Technical Report, 2016.
% function [U, V] = fm_train(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test)
% Inputs:
%   R: rating matrix
%   U, V: the interaction (d-by-n) matrices.
%   U_reg, V_reg: the frequncy-aware regularization coefficients of the two interaction matrices.
%   epsilon: stopping tolerance in (0,1). Use a larger value if the training time is too long.
%   R_test: testing rating matrix
% Outputs:
%   U, V: the interaction (d-by-n) matrices.

    [m, n] = size(R);
    nnz_R_test = nnz(R_test);
    alpha = 120;
    IR = spones(R); 
    P = full(IR);% p_ui = preference of user u to item i
    C = IR+alpha*R;% conﬁdence in observing p_ui
    C_minus_I = alpha*R;
    eye_d = eye(d);
%    eye_m = eye(m);
%    eye_n = eye(n);

    fprintf('%4s  %15s  %15s  %15s  %15s\n', 'iter', 'time', 'obj', 'test_loss', 'loss');
    for k = 1:max_iter
        time1=tic;
        VVT = V*V';
        for i = 1:m
            U(:,i) = inv(VVT+U_reg(i)*eye(d))*V*R(i,:)';
%            U(:,i) = inv(VVT+V*(diag(C(i,:))-eye_n)*V'+U_reg(i)*eye_d)*V*diag(C(i,:))*P(i,:)';
%            U(:,i) = inv(VVT+V*diag(C_minus_I(i,:))*V'+U_reg(i)*eye_d)*V*diag(C(i,:))*R(i,:)';
%            U(:,i) = inv(V*diag(C(i,:))*V'+U_reg(i)*eye_d)*V*diag(C(i,:))*R(i,:)';
        end
        UUT = U*U';
        for i = 1:n
            V(:,i) = inv(UUT+V_reg(i)*eye(d))*U*R(:,i);
%            V(:,i) = inv(UUT+U*(diag(C(:,i))-eye_m)*U'+V_reg(i)*eye_d)*U*diag(C(:,i))*P(:,i);
%            V(:,i) = inv(UUT+U*diag(C_minus_I(:,i))*U'+V_reg(i)*eye_d)*U*diag(C(:,i))*R(:,i);
%            V(:,i) = inv(U*diag(C(:,i))*U'+V_reg(i)*eye_d)*U*diag(C(:,i))*R(:,i);
        end
        time2=toc(time1);

        Y_test_tilde = get_embedding_inner(U,V,R_test);
        test_loss = sqrt(full(sum(sum((R_test-Y_test_tilde).*(R_test-Y_test_tilde)))/nnz_R_test));
        B = get_embedding_inner(U, V, R)-R;
        loss = 0.5 * full(sum(sum(B .* B)));
%        c_loss = 0.5 * full(sum(sum((B .* B) .* C)));
        f = 0.5*(sum(U.*U)*U_reg+sum(V.*V)*V_reg)+loss;
%        f = 0.5*(sum(U.*U)*U_reg+sum(V.*V)*V_reg)+c_loss;
        
        fprintf('%4d  %15.3f  %15.3f  %15.6f  %15.3f\n', k, time2, f, test_loss, loss);
    end
end
%point wise summation
% z_(m,n) = u_m^T*v_n
function Z = get_embedding_inner(U, V, R)
    [m, n] = size(R);
    [i_idx, j_idx, vals] = find(R);
    l = nnz(R);
    num_batches = 10;
    bsize = ceil(l/num_batches);
    for i = 1: num_batches
        range = (i - 1) * bsize + 1 : min(l, i * bsize);
        vals(range) = dot( V(:, j_idx(range)), U(:, i_idx(range)) );
    end
    Z = sparse(i_idx, j_idx, vals, m, n);
end
