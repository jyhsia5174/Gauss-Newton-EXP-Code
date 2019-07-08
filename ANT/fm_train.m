function [U, V] = fm_train(y, W, H, U_reg, V_reg, d, epsilon, do_pcond)
% Train a factorization machine using the proposed method in the paper below.
%   Wei-Sheng Chin, Bo-Wen Yuan, Meng-Yuan Yang, and Chih-Jen Lin, An Efficient Alternating Newton Method for Learning Factorization Machines, Technical Report, 2016.
% function [w, U, V] = fm_train(y, X, lambda_w, lambda_U, lambda_V, d, epsilon, do_pcond, sub_rate)
% Inputs:
%   y: training labels, an l-dimensional binary vector. Each element should be either +1 or -1.
%   X: training instances. X is an l-by-n matrix if you have l training instances in an n-dimensional feature space.
%   lambda_w: the regularization coefficient of linear term.
%   lambda_U, lambda_V: the regularization coefficients of the two interaction matrices.
%   d: dimension of the latent space.
%   epsilon: stopping tolerance in (0,1). Use a larger value if the training time is too long.
%   do_pcond: a flag. Use 1/0 to enable/disable the diagonal preconditioner.
%   sub_rate: sampling rate in (0,1] to select instances for the sub-sampled Hessian matrix.
% Outputs:
%   w: linear coefficients. An n-dimensional vector.
%   U, V: the interaction (d-by-n) matrices.
    tic;
    max_iter = 1000;

    [l, m] = size(W);
    [l, n] = size(H);

    rand('seed', 0);

    U = 2*(0.1/sqrt(d))*(rand(d,m)-0.5);
    V = 2*(0.1/sqrt(d))*(rand(d,n)-0.5);

    y_tilde = (sum((U*W').*(V*H'),1))';

    b = y_tilde-y;
    loss = 0.5*sum(b.*b);

    f = 0.5*(sum(U.*U)*U_reg+sum(V.*V)*V_reg)+loss;
    G_norm_0 = 0;

    fprintf('iter        time              obj          |grad|           |gradU| (#nt,#cg)           |gradV| (#nt,#cg)\n');
    for k = 1:max_iter
        [U, y_tilde, b, f, loss, nt_iters_U, G_norm_U, cg_iters_U] = update_block(y, W, U, V*H', y_tilde, b, f, loss, U_reg, do_pcond);
        [V, y_tilde, b, f, loss, nt_iters_V, G_norm_V, cg_iters_V] = update_block(y, H, V, U*W', y_tilde, b, f, loss, V_reg, do_pcond);
        G_norm = norm([G_norm_U, G_norm_V]);
        if (k == 1)
            G_norm_0 = G_norm;
        end
        if (G_norm <= epsilon*G_norm_0)
            break;
        end
        fprintf('%4d  %11.3f  %14.6f  %14.6f    %14.6f (%3d,%3d)    %14.6f (%3d,%3d)\n', k, toc, f, G_norm, G_norm_U, nt_iters_U, cg_iters_U, G_norm_V, nt_iters_V, cg_iters_V);
        if (k == max_iter)
            fprintf('Warning: reach max training iteration. Terminate training process.\n');
        end
    end
end

% See Algorithm 3 in the paper. 
function [U, y_tilde, b, f, loss, nt_iters, G_norm, total_cg_iters] = update_block(y, W, U, Q, y_tilde, b, f, loss, lambda_freq, do_pcond)
    epsilon = 0.8;
    max_nt_iter = 100;
    l = size(W,1);
    m = size(U,2);
    G0_norm = 0;
    total_cg_iters = 0;
    nt_iters = 0;
    for k = 1:max_nt_iter
        G = U*sparse([1:m], [1:m], lambda_freq)+Q*sparse([1:l], [1:l], b)*W;
        G_norm = sqrt(sum(sum(G.*G)));
        if (k == 1)
            G0_norm = G_norm;
        end
        if (G_norm <= epsilon*G0_norm)
            return;
        end
        nt_iters = k;
        if (k == max_nt_iter)
            fprintf('Warning: reach newton iteration bound before gradient norm is shrinked enough.\n');
        end
        [S, cg_iters] = pcg(W, Q, G, lambda_freq);
        total_cg_iters = total_cg_iters+cg_iters;

        Delta = (sum(Q'.*(W*S'),2));
        US = sum(U.*S); SS = sum(S.*S);
        y_tilde = y_tilde+Delta;
        b = y_tilde - y;
        loss_new = 0.5*sum(b .* b);
        f_diff = 0.5*(2*(US+SS)*lambda_freq)+loss_new-loss;
        loss = loss_new;
        f = f+f_diff;
        U = U+S;
    end
end

% See Algorithm 4 in the paper.
function [S, cg_iters] = pcg(W, Q, G, lambda_freq)
    zeta = 1e-2;
    cg_max_iter = 100;
    l = size(W,1);
    m = size(G,2);
    s_bar = zeros(size(G));
    r = -G;
    d = r;
    G0G0 = sum(sum(r.*r));
    gamma = G0G0;
    cg_iters = 0;
    while (gamma > zeta*zeta*G0G0)
        cg_iters = cg_iters+1;
        z = sum(Q'.*(W*d'),2);
        Dh = d*sparse([1:m], [1:m], lambda_freq)+Q*sparse([1:l], [1:l], z)*W;
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
    S = s_bar;
end
