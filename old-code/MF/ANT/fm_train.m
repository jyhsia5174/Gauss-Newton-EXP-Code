function [U, V] = fm_train(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test)
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
    total_t=0;

    fprintf('%4s  %15s  %3s  %3s  %15s  %15s  %15s  %15s  %15s  %15s\n', 'iter', 'time', '#cg_U', '#cg_V', 'obj', '|G_U|', '|G_V|', 'test_loss', '|G|', 'loss');
    for k = 1:max_iter
        if (k == 1)
            B = get_embedding_inner(U, V, R, i_idx_R, j_idx_R)-R;
            loss = 0.5 * full(sum(sum(B .* B)));
            freq_reg = 0.5*(sum(U.*U)*U_reg+sum(V.*V)*V_reg);
            f = freq_reg+loss;
            GU = U.*U_reg'+V*B';
            GV = V.*V_reg'+U*B;
            G_norm = norm([GU GV],'fro');
            G_norm_0 = G_norm;
            fprintf('initial G_norm: %15.6f\n', G_norm_0);
            fprintf('initial reg: %15.6f\n', freq_reg);
            fprintf('initial loss: %15.6f\n', loss); 
        end

        time1=tic;
        GU = U.*U_reg'+V*B';
        [U, B, f, loss, cg_iters_U] = update_block(U, V, B, R, GU, f, loss, U_reg, 'no_transposed', i_idx_R, j_idx_R);
        GV = V.*V_reg'+U*B;
        [V, B, f, loss, cg_iters_V] = update_block(V, U, B, R, GV, f, loss, V_reg, 'transposed', i_idx_R, j_idx_R);
        time2=toc(time1);
        total_t=total_t+time2;

        Y_test_tilde = get_embedding_inner(U,V,R_test, i_idx_R_test, j_idx_R_test);
        test_loss = sqrt(full(sum(sum((R_test-Y_test_tilde).*(R_test-Y_test_tilde)))/nnz_R_test));
        GU = U.*U_reg'+V*B';
        GV = V.*V_reg'+U*B;
        G_norm = norm([GU GV],'fro');
        G_norm_U = norm(GU,'fro');
        G_norm_V = norm(GV,'fro');
        fprintf('%4d  %15.3f  %3d  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, total_t, cg_iters_U, cg_iters_V, f, G_norm_U, G_norm_V, test_loss, G_norm, loss);

        if (G_norm <= epsilon*G_norm_0)
            break;
        end
    end
    if (k == max_iter)
        fprintf('Warning: reach max training iteration. Terminate training process.\n');
    end
end
function [U, B, f, loss, cg_iters] = update_block(U, V, B, R, G, f, loss, reg, option, i_idx_R, j_idx_R)
    eta = 0.3;
    cg_max_iter = 20;
    Su = zeros(size(G));
    C = -G;
    D = C;
    gamma_0 = sum(sum(C.*C));
    gamma = gamma_0;
    cg_iters = 0;
    while (gamma > eta*eta*gamma_0)
        cg_iters = cg_iters+1;
        if strcmp(option,'transposed')
            Z = get_embedding_inner(V, D, R, i_idx_R, j_idx_R);
            Dh = D.*reg'+V*Z;
        else
            Z = get_embedding_inner(D, V, R, i_idx_R, j_idx_R);
            Dh = D.*reg'+V*Z';
        end
        alpha = gamma/sum(sum(D.*Dh));
        Su = Su+alpha*D;
        C = C-alpha*Dh;
        gamma_new = sum(sum(C.*C));
        beta = gamma_new/gamma;
        D = C+beta*D;
        gamma = gamma_new;
        if (cg_iters >= cg_max_iter)
            fprintf('Warning: reach max CG iteration. CG process is terminated.\n');
            break;
        end
    end
    if strcmp(option,'transposed')
        Delta = get_embedding_inner(V, Su, R, i_idx_R, j_idx_R);
    else
        Delta = get_embedding_inner(Su, V, R, i_idx_R, j_idx_R);
    end
    B_new = B+Delta;
    USu = sum(U.*Su);
    SuSu = sum(Su.*Su);
    loss_new = 0.5*full(sum(sum(B_new.*B_new)));
    f_diff = 0.5*((2*USu+SuSu)*reg)+loss_new-loss;
    f = f+f_diff;
    U = U+Su;
    B = B_new;
    loss = loss_new;
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
