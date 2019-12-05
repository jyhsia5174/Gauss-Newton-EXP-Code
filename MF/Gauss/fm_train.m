function [U, V] = fm_train(y, W, H, U_reg, V_reg, d, epsilon, max_iter, do_pcond, y_test, W_test, H_test)
% Train a factorization machine using the proposed method in the paper below.
%   Wei-Sheng Chin, Bo-Wen Yuan, Meng-Yuan Yang, and Chih-Jen Lin, An Efficient Alternating Newton Method for Learning Factorization Machines, Technical Report, 2016.
% function [w, U, V] = fm_train(y, X, lambda, d, epsilon, do_pcond, sub_rate)
% Inputs:
%   y: training labels, an l-dimensional binary vector. Each element should be either +1 or -1.
%   X: training instances. X is an l-by-n matrix if you have l training instances in an n-dimensional feature space.
%   lambda: the regularization coefficients of the two interaction matrices.
%   d: dimension of the latent space.
%   epsilon: stopping tolerance in (0,1). Use a larger value if the training time is too long.
%   do_pcond: a flag. Use 1/0 to enable/disable the diagonal preconditioner.
%   sub_rate: sampling rate in (0,1] to select instances for the sub-sampled Hessian matrix.
% Outputs:
%   w: linear coefficients. An n-dimensional vector.
%   U, V: the interaction (d-by-n) matrices.
    tic;
%    max_iter = 100;

    [l, m] = size(W);
    [l, n] = size(H);
    IR = W'*H;
    Y = init_Y(W, H, y);

    rand('seed', 0);

    U = 2*(0.1/sqrt(d))*(rand(d,m)-0.5);
    V = 2*(0.1/sqrt(d))*(rand(d,n)-0.5);

    Y_tilde = get_embedding_inner(U, V, IR);
    B = Y_tilde - Y;
    loss = 0.5 * full(sum(sum(B .* B)));

    f = 0.5*(sum(U.*U)*U_reg+sum(V.*V)*V_reg)+loss;
    G_norm_0 = 0; G_norm = 0;

    fprintf('%4s  %15s  %3s  %15s  %15s  %15s  %15s  %15s  %15s\n', 'iter', 'time', '#cg', 'obj', '|grad|', 'va_loss', '|GV|', '|GV|', 'loss');
    for k = 1:max_iter
        
        Y_tilde = get_embedding_inner(U, V, IR);
        B = Y_tilde - Y;
        G = [U*sparse([1:m], [1:m], U_reg) V*sparse([1:n], [1:n], V_reg)] + [V*((B.*IR)') U*(B.*IR)];


        G_norm = sqrt(sum(sum(G.*G)));
        if (k == 1)
            G_norm_0 = G_norm;
            fprintf('Warning: %15.6f\n', G_norm_0);
        end

		if (G_norm <= epsilon*G_norm_0)
			break;
		end

        [U, V, Y_tilde, B, f, loss, cg_iters] = update(Y, W, H, U, V, Y_tilde, B, IR, f, loss, U_reg, V_reg, G);

        y_test_tilde = fm_predict( W_test, H_test, U, V);
        va_loss = mean((y_test - y_test_tilde) .* (y_test - y_test_tilde));

        Y_tilde = get_embedding_inner(U, V, IR);
        B = Y_tilde - Y;
        G = [U*sparse([1:m], [1:m], U_reg) V*sparse([1:n], [1:n], V_reg)] + [V*((B.*IR)') U*(B.*IR)];

        G_norm = sqrt(sum(sum(G.*G)));
        GU_norm = sqrt(sum(sum(G(1:end, 1:m).*G(1:end, 1:m))));
        GV_norm = sqrt(sum(sum(G(1:end, m+1:end).*G(1:end, m+1:end))));
        fprintf('%4d  %15.3f  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, toc, cg_iters, f, G_norm, va_loss, GU_norm, GV_norm, loss);

        if (k == max_iter)
            fprintf('Warning: reach max training iteration. Terminate training process.\n');
        end
    end
end

% See Algorithm 3 in the paper.
function [U, V, Y_tilde, B, f, loss, total_cg_iters] = update(Y, W, H, U, V, Y_tilde, B, IR, f, loss, U_reg, V_reg, G)
    epsilon = 0.8;
    nu = 0.1;
    min_step_size = 1e-20;
    l = size(W,1); m = size(U,2); n = size(V,2);
    total_cg_iters = 0;

    [Su, Sv, cg_iters] = cg(W, H, U, V, IR, G, U_reg, V_reg);
    total_cg_iters = total_cg_iters+cg_iters;

    WS_u = (Su*W');
    HS_v = (Sv*H');
    Delta_1 = get_cross_embedding_inner(Su, Sv, U, V, IR);
    Delta_2 = get_embedding_inner(Su, Sv, IR);

    US_u = sum(U.*Su)*U_reg; VS_v = sum(V.*Sv)*V_reg;
    SS = sum([Su Sv].*[Su Sv])*[U_reg ; V_reg];
	GS = sum(sum(G.*[Su Sv]));
    theta = 1;
    while (true)
        if (theta < min_step_size)
            fprintf('Warning: step size is too small in line search. Switch to the next block of variables.\n');
            return;
        end
        Y_tilde_new = Y_tilde+theta*Delta_1+theta*theta*Delta_2;
        B_new = Y_tilde_new-Y;
        loss_new = 0.5*full(sum(sum(B_new.*B_new)));
        f_diff = 0.5*(2*theta*(US_u+VS_v)+theta*theta*SS)+loss_new-loss;
        if (f_diff <= nu*theta*GS)
            loss = loss_new;
            f = f+f_diff;
            U = U+theta*Su;
            V = V+theta*Sv;
            Y_tilde = Y_tilde_new;
            B = B_new;
            break;
        end
        theta = theta*0.5;
    end
    if (theta ~= 1)
        fprintf('Warning: Doing line search %14.10f\n', theta);
    end

end

% See Algorithm 4 in the paper.
function [Su, Sv, cg_iters] = cg(W, H, U, V, IR, G, U_reg, V_reg)
    zeta = 0.3;
    cg_max_iter = 20;
    [l, m] = size(W);
    s_bar = zeros(size(G));
    r = -G;
    d = r;
    G0G0 = sum(sum(r.*r));
    gamma = G0G0;
    cg_iters = 0;
    lambda_freq = sparse([1:size(G,2)], [1:size(G,2)], [U_reg ; V_reg]);
    while (gamma > zeta*zeta*G0G0)
        cg_iters = cg_iters+1;

        [Z] = get_cross_embedding_inner(d(:,1:m), d(:,m+1:end), U, V, IR);
        Dh = d*lambda_freq + [V*((Z.*IR)') U*(Z.*IR)];
        alpha = gamma/sum(sum(d.*Dh));
        s_bar = s_bar+alpha*d;
        r = r-alpha*Dh;
        gamma_new = sum(sum(r.*r));
        beta = gamma_new/gamma;
        d = r+beta*d;
        gamma = gamma_new;
        if (cg_iters >= cg_max_iter)
            fprintf('Warning: reach max CG iteration. CG process is terminated.\n');
            break;
        end
    end
    Su = s_bar(1:end, 1:m);
    Sv = s_bar(1:end, m+1:end);
end

% (H*V')./(W*Su')+(W*U')./(H*Sv')
function [Z] = get_cross_embedding_inner(Su, Sv, U, V, IR)
    [m, n] = size(IR);
    nnz_num = nnz(IR);
    z_i = {}; z_j = {}; z_val = {};
    parfor j = 1:n
        [i_idxs, j_idxs, dummy] = find(IR(:, j));
        vals = V(:,j)'*Su(:,i_idxs) + Sv(:,j)'*U(:,i_idxs);
        j_idxs(1:end) = j;
        z_i{j} = i_idxs;
        z_j{j} = j_idxs;
        z_val{j} = vals';
    end
    Z_i = cat(1, z_i{:});
    Z_j = cat(1, z_j{:});
    Z_val = cat(1, z_val{:});
    Z = sparse(Z_i, Z_j, Z_val, m, n);
end

% (W*Su) ./ (H*Sv)
function [Z] = get_embedding_inner(U, V, IR)
    [m, n] = size(IR);
    nnz_num = nnz(IR);
    z_i = {}; z_j = {}; z_val = {};
    parfor j = 1:n
        [i_idxs, j_idxs, dummy] = find(IR(:, j));
        vals = V(:,j)'*U(:,i_idxs);
        j_idxs(1:end) = j;
        z_i{j} = i_idxs;
        z_j{j} = j_idxs;
        z_val{j} = vals';
    end
    Z_i = cat(1, z_i{:});
    Z_j = cat(1, z_j{:});
    Z_val = cat(1, z_val{:});
    Z = sparse(Z_i, Z_j, Z_val, m, n);
end

% (W*Su) ./ (H*Sv)
function [Y] = init_Y(W, H, y)
    [l, m] = size(W);
    [l, n] = size(H);
    [wi, wj, wv] = find(W);
    [hi, hj, hv] = find(H);
    wij = sortrows(cat(2,wi, wj));
    hij = sortrows(cat(2,hi, hj));
    Y = sparse(wij(:, 2), hij(:, 2), y, m, n);
end

%function [Z] = get_cross_embedding_inner_mat_v2(Su, Sv, U, V, IR)
%    tic;
%    [m, n] = size(IR);
%    nnz_num = nnz(IR);
%    z_i = {}; z_j = {}; z_val = {};
%    parfor i = 1:m
%        [i_idxs, j_idxs, vals] = find(IR(i, :));
%        vals = V(:,j_idxs)'*Su(:,i) + Sv(:,j_idxs)'*U(:,i);
%        z_i{i} = i_idxs;
%        z_j{i} = j_idxs;
%        z_val{i} = vals';
%    end
%    Z_i = cat(2, z_i{:});
%    Z_j = cat(2, z_j{:});
%    Z_val = cat(2, z_val{:});
%    Z = sparse(Z_i, Z_j, Z_val, m, n);
%    fprintf('Time get_cross_embedding_inner_mat %f\n', toc);
%end

%function [Z] = get_cross_embedding_inner(Su, Sv, U, V, IR)
%    tic;
%    [m, n] = size(IR);
%    nnz_num = nnz(IR);
%    [i_idxs, j_idxs, vals] = find(IR);
%    parfor k = 1:nnz_num
%        i = i_idxs(k);
%        j = j_idxs(k);
%        vals(k) = V(:,j)'*Su(:,i) + U(:,i)'*Sv(:,j);
%    end
%    Z = sparse(i_idxs, j_idxs, vals, m, n);
%    fprintf('Time get_cross_embedding_inner %f\n', toc);
%end

%function [Z] = get_cross_embedding_inner_mat(Su, Sv, U, V, IR)
%    tic;
%    [m, n] = size(IR);
%    nnz_num = nnz(IR);
%    z_i = []; z_j = []; z_val = [];
%    for i = 1:m
%        [dummy, j_idxs, dummy] = find(IR(i, :));
%        vals = V(:,j_idxs)'*Su(:,i) + Sv(:,j_idxs)'*U(:,i);
%        i_idxs = j_idxs;
%        i_idxs(1:end) = i;
%        z_i = [z_i i_idxs];
%        z_j = [z_j j_idxs];
%        z_val = [z_val vals'];
%    end
%    Z = sparse(z_i, z_j, z_val, m, n);
%    fprintf('Time get_cross_embedding_inner_mat %f\n', toc);
%end

