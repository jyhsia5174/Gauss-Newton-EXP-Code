function [w, U, V] = fm_train(y, X, lambda_w, lambda_U, lambda_V, d, epsilon, do_pcond, sub_rate)
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
    [l, n] = size(X);
    w = zeros(1,n);
    rand('seed', 0);
    U = 2*(0.1/sqrt(d))*(rand(d,n)-0.5);
    V = 2*(0.1/sqrt(d))*(rand(d,n)-0.5);
    y_tilde = X*w'+0.5*(sum((U*X').*(V*X'),1))';
    expyy = exp(y.*y_tilde);
    loss = sum(log1p(1./expyy));
    f = 0.5*(lambda_w*sum(w.*w)+lambda_U*sum(sum(U.*U))+lambda_V*sum(sum(V.*V)))+loss;
    G_norm_0 = 0;
    fprintf('iter        time              obj          |grad|           |gradw| (#nt,#cg)           |gradU| (#nt,#cg)           |gradV| (#nt,#cg)\n');
    for k = 1:max_iter
        [w, y_tilde, expyy, f, loss, nt_iters_w, G_norm_w, cg_iters_w] = update_block(y, X, w, 2*ones(1,l), y_tilde, expyy, f, loss, lambda_w, do_pcond, sub_rate);
        [U, y_tilde, expyy, f, loss, nt_iters_U, G_norm_U, cg_iters_U] = update_block(y, X, U, V*X', y_tilde, expyy, f, loss, lambda_U, do_pcond, sub_rate);
        [V, y_tilde, expyy, f, loss, nt_iters_V, G_norm_V, cg_iters_V] = update_block(y, X, V, U*X', y_tilde, expyy, f, loss, lambda_V, do_pcond, sub_rate);
        G_norm = norm([G_norm_w, G_norm_U, G_norm_V]);
        if (k == 1)
            G_norm_0 = G_norm;
        end
        if (G_norm <= epsilon*G_norm_0)
            break;
        end
        fprintf('%4d  %11.3f  %14.6f  %14.6f    %14.6f (%3d,%3d)    %14.6f (%3d,%3d)    %14.6f (%3d,%3d)\n', k, toc, f, G_norm, G_norm_w, nt_iters_w, cg_iters_w, G_norm_U, nt_iters_U, cg_iters_U, G_norm_V, nt_iters_V, cg_iters_V);
        if (k == max_iter)
            fprintf('Warning: reach max training iteration. Terminate training process.\n');
        end
    end
end

% See Algorithm 3 in the paper. 
function [U, y_tilde, expyy, f, loss, nt_iters, G_norm, total_cg_iters] = update_block(y, X, U, Q, y_tilde, expyy, f, loss, lambda, do_pcond, sub_rate)
    epsilon = 0.8;
    nu = 0.1;
    max_nt_iter = 100;
    min_step_size = 1e-20;
    l = size(X,1);
    G0_norm = 0;
    total_cg_iters = 0;
    nt_iters = 0;
    for k = 1:max_nt_iter
        G = lambda*U+0.5*Q*sparse([1:l], [1:l], -y./(1+expyy))*X;
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
        D = sparse([1:l], [1:l], expyy./(1+expyy)./(1+expyy));
        [S, cg_iters] = pcg(X, Q, G, D, lambda, do_pcond, sub_rate);
        total_cg_iters = total_cg_iters+cg_iters;
        Delta = 0.5*(sum(Q'.*(X*S'),2));
        US = sum(sum(U.*S)); SS = sum(sum(S.*S)); GS = sum(sum(G.*S));
        theta = 1;
        while (true)
            if (theta < min_step_size)
                fprintf('Warning: step size is too small in line search. Switch to the next block of variables.\n');
                return;
            end
            y_tilde_new = y_tilde+theta*Delta;
            expyy_new = exp(y.*y_tilde_new);
            loss_new = sum(log1p(1./expyy_new));
            f_diff = 0.5*lambda*(2*theta*US+theta*theta*SS)+loss_new-loss;
            if (f_diff <= nu*theta*GS)
                loss = loss_new;
                f = f+f_diff;
                U = U+theta*S;
                y_tilde = y_tilde_new;
                expyy = expyy_new;
                break;
            end
            theta = theta*0.5;
        end
    end
end

% See Algorithm 4 in the paper.
function [S, cg_iters] = pcg(X, Q, G, D, lambda, do_pcond, sub_rate)
    zeta = 0.3;
    cg_max_iter = 100;
    if (sub_rate < 1)
        l = size(X,1);
        whole = randperm(l);
        selected = sort(whole(1:max(1, floor(sub_rate*l))));
        X = X(selected,:);
        Q = Q(:,selected);
        D = D(selected,selected);
    end
    l = size(X,1);
    s_bar = zeros(size(G));
    M = ones(size(G));
    if (do_pcond)
        M = 1./sqrt(lambda+(1/sub_rate)*0.25*(Q.*Q)*(D*(X.*X)));
    end
    r = -M.*G;
    d = r;
    G0G0 = sum(sum(r.*r));
    gamma = G0G0;
    cg_iters = 0;
    while (gamma > zeta*zeta*G0G0)
        cg_iters = cg_iters+1;
        Dh = M.*d;
        z = 0.5*sum(Q'.*(X*Dh'),2);
        Dh = M.*(lambda*Dh+0.5*(1/sub_rate)*Q*sparse([1:l], [1:l], D*z)*X);
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
    S = M.*s_bar;
end
