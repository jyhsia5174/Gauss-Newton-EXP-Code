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
    [i_idx_R, j_idx_R, vals_R] = find(R);
    [i_idx_R_test, j_idx_R_test, vals_R_test] = find(R_test);
    uni_i_idx_R = unique(i_idx_R);
    uni_j_idx_R = unique(j_idx_R);   
    
    U = U(1:d,:);
    V = V(1:d,:);

    for i = 1:length(uni_i_idx_R)
        m2ns{uni_i_idx_R(i)}=find(R(uni_i_idx_R(i),:));
    end

    for i = 1:length(uni_j_idx_R)
        n2ms{uni_j_idx_R(i)}=find(R(:,uni_j_idx_R(i))');
    end

    B = get_embedding_inner(U, V, R, i_idx_R, j_idx_R)-R;
    loss = 0.5 * full(sum(sum(B .* B)));
    freq_reg = 0.5*(sum(U.*U)*U_reg+sum(V.*V)*V_reg);
    fprintf('initial reg: %15.6f\n', freq_reg);
    fprintf('initial loss: %15.6f\n', loss); 

    fprintf('%4s  %15s  %15s  %15s  %15s\n', 'iter', 'time', 'obj', 'test_loss', 'loss');
    total_t=0;
    for k = 1:max_iter
        time1=tic;
        U = updata_block(U,V,R,uni_i_idx_R,U_reg,d,m2ns);
        V = updata_block(V,U,R',uni_j_idx_R,V_reg,d,n2ms);
        time2=toc(time1);
        total_t=total_t+time2;

        Y_test_tilde = get_embedding_inner(U,V,R_test,i_idx_R_test, j_idx_R_test);
        test_loss = sqrt(full(sum(sum((R_test-Y_test_tilde).*(R_test-Y_test_tilde))))/nnz_R_test);
        B = get_embedding_inner(U, V, R, i_idx_R, j_idx_R)-R;
        loss = 0.5 * full(sum(sum(B .* B)));
        f = 0.5*(sum(U.*U)*U_reg+sum(V.*V)*V_reg)+loss;
        
        fprintf('%4d  %15.3f  %15.3f  %15.6f  %15.3f\n', k, total_t, f, test_loss, loss);
    end
end
%point wise summation
% z_(m,n) = u_m^T*v_n
function Z = get_embedding_inner(U, V, R, i_idx, j_idx)
    [m, n] = size(R);
    l = nnz(R);
    vals = zeros(1,l);
    num_batches = 10;
    bsize = ceil(l/num_batches);
    for i = 1: num_batches
        range = (i - 1) * bsize + 1 : min(l, i * bsize);
        vals(range) = dot( V(:, j_idx(range)), U(:, i_idx(range)) );
    end
    Z = sparse(i_idx, j_idx, vals, m, n);
end

function U = updata_block(U,V,R,uni_i_idx_R,U_reg,d,m2ns)
    temp = zeros(d,length(uni_i_idx_R));
    parfor i = 1:length(uni_i_idx_R)
        ii = uni_i_idx_R(i);
        idx=m2ns{ii};
        VVT = V(:,idx)*V(:,idx)';
        temp(:,i) = inv(VVT+U_reg(ii)*eye(d))*V*R(ii,:)';
    end
    U(:,uni_i_idx_R) = temp;
end
