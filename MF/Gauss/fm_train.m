function [U, V] = fm_train(R, IR, U_reg, V_reg, d, epsilon, max_iter, R_test, IR_test)
% function [U, V] = fm_train(R, IR, U_reg, V_reg, d, epsilon, max_iter, R_test, IR_test)
% Inputs:
%   R: rating matrix
%   IR: each entry is result of indicator function from corresponding R matrix.
%   U_reg, V_reg: the frequncy-aware regularization coefficients of the two interaction matrices.
%   d: dimension of the latent space.
%   epsilon: stopping tolerance in (0,1). Use a larger value if the training time is too long.
%   R_test: testing rating matrix
%   IR_test: each entry is result of indicator function from corresponding R_test matrix.
% Outputs:
%   U, V: the interaction (d-by-n) matrices.
    tic;

    [m, n] = size(R);
    nnz_R_test = nnz(R_test);

    rand('seed', 0);

    U = 2*(0.1/sqrt(d))*(rand(d,m)-0.5);
    V = 2*(0.1/sqrt(d))*(rand(d,n)-0.5);

    nu = 0.1;
    min_step_size = 1e-20;

    fprintf('%4s  %15s  %3s  %3s  %15s  %15s  %15s  %15s  %15s\n', 'iter', 'time', '#cg', '#ls', 'obj', '|G|', 'test_loss', '|G_V|', '|G_V|', 'loss');
    for k = 1:max_iter
        if (k == 1)
            B = get_embedding_inner(U, V, IR) - R;
            loss = 0.5 * full(sum(sum(B .* B)));
            G = [U*spdiags(U_reg,0,m,m) V*spdiags(V_reg,0,n,n)] + [V*((B.*IR)') U*(B.*IR)];
            f = 0.5*(sum(U.*U)*U_reg+sum(V.*V)*V_reg)+loss;
            G_norm = norm(G,'fro');
            G_norm_0 = G_norm;
            fprintf('initial G_norm: %15.6f\n', G_norm_0);
        end

        if (G_norm <= epsilon*G_norm_0)
            break;
        end

        [Su, Sv, cg_iters] = cg(U, V, IR, G, U_reg, V_reg);

        Delta_1 = get_cross_embedding_inner(Su, Sv, U, V, IR);
        Delta_2 = get_embedding_inner(Su, Sv, IR);
        US_u = sum(U.*Su)*U_reg; VS_v = sum(V.*Sv)*V_reg;
        SS = sum([Su Sv].*[Su Sv])*[U_reg ; V_reg];
        GS = sum(sum(G.*[Su Sv]));
        theta = 1;
        for ls_steps = 1:intmax;
            if (theta < min_step_size)
                fprintf('Warning: step size is too small in line search. Switch to the next block of variables.\n');
                return;
            end
            B_new = B+theta*Delta_1+theta*theta*Delta_2;
            loss_new = 0.5*full(sum(sum(B_new.*B_new)));
            f_diff = 0.5*(2*theta*(US_u+VS_v)+theta*theta*SS)+loss_new-loss;
            if (f_diff <= nu*theta*GS)
                loss = loss_new;
                f = f+f_diff;
                U = U+theta*Su;
                V = V+theta*Sv;
                B = B_new;
                 break;
            end
            theta = theta*0.5;
        end

        Y_test_tilde = get_embedding_inner(U,V,IR_test);
        test_loss = full(sum(sum((R_test-Y_test_tilde).*(R_test-Y_test_tilde)))/nnz_R_test);

        G = [U*spdiags(U_reg,0,m,m) V*spdiags(V_reg,0,n,n)] + [V*((B.*IR)') U*(B.*IR)];
        G_norm = norm(G,'fro');
        GU_norm = norm(G(:, 1:m),'fro');
        GV_norm = norm(G(:, m+1:end),'fro');

        fprintf('%4d  %15.3f  %3d  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, toc, cg_iters, ls_steps, f, G_norm, test_loss, GU_norm, GV_norm, loss);
    end
    if (k == max_iter)
        fprintf('Warning: reach max training iteration. Terminate training process.\n');
    end

end

% See Algorithm 4 in the paper.
function [Su, Sv, cg_iters] = cg(U, V, IR, G, U_reg, V_reg)
    eta = 0.3;
    cg_max_iter = 20;
    [m, n] = size(IR);
    S = zeros(size(G));
    C = -G;
    D = C;
    gamma_0 = sum(sum(C.*C));
    gamma = gamma_0;
    cg_iters = 0;
    reg = spdiags([U_reg ; V_reg], 0, size(G,2), size(G,2));
    while (gamma > eta*eta*gamma_0)
        cg_iters = cg_iters+1;
        Z = get_cross_embedding_inner(D(:,1:m), D(:,m+1:end), U, V, IR);
        Dh = D*reg + [V*((Z.*IR)') U*(Z.*IR)];
        alpha = gamma/sum(sum(D.*Dh));
        S = S+alpha*D;
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
    Su = S(:, 1:m);
    Sv = S(:, m+1:end);
end

% (H*V')./(W*Su')+(W*U')./(H*Sv')
function [Z] = get_cross_embedding_inner(Su, Sv, U, V, IR)
    [m, n] = size(IR);
    [i_idx, j_idx, vals] = find(IR);
    l = nnz(IR);
    num_batches = 10;
    bsize = ceil(l/num_batches);
    for i = 1: num_batches
        range = (i - 1) * bsize + 1 : min(l, i * bsize);
        vals(range) = dot( V(:, j_idx(range)) ,Su(:, i_idx(range))) + dot(Sv(:, j_idx(range)),U(:, i_idx(range)));
    end
    Z = sparse(i_idx, j_idx, vals, m, n);
end

%% (W*Su) ./ (H*Sv)
function [Z] = get_embedding_inner(U, V, IR)
    [m, n] = size(IR);
    [i_idx, j_idx, vals] = find(IR);
    l = nnz(IR);
    num_batches = 10;
    bsize = ceil(l/num_batches);
    for i = 1: num_batches
        range = (i - 1) * bsize + 1 : min(l, i * bsize);
        vals(range) = dot( V(:, j_idx(range)), U(:, i_idx(range)) );
    end
    Z = sparse(i_idx, j_idx, vals, m, n);
end

