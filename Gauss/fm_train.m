function [U, V] = fm_train(y, W, H, lambda, d, epsilon, do_pcond)
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
    max_iter = 1000;

    [l, m] = size(W);
    [l, n] = size(H);

    rand('seed', 0);

    U = 2*(0.1/sqrt(d))*(rand(d,m)-0.5);
    V = 2*(0.1/sqrt(d))*(rand(d,n)-0.5);

    y_tilde = (sum((U*W').*(V*H'),1))';
    b = y_tilde-y;
    loss = 0.5*sum(b.*b);

    f = 0.5*lambda*(sum(sum(U.*U))+sum(sum(V.*V)))+loss;
    G_norm_0 = 0;

    fprintf('iter        time              obj          |grad|          #cg\n');
    for k = 1:max_iter
        [U, V, y_tilde, b, f, loss, nt_iters, G_norm, cg_iters] = update(y, W, H, U, V, U*W', V*H', y_tilde, b, f, loss, lambda);
        if (k == 1)
            G_norm_0 = G_norm;
        end
        if (G_norm <= epsilon*G_norm_0)
            break;
        end
        fprintf('%4d  %11.3f  %14.6f  %14.6f %3d\n', k, toc, f, G_norm, cg_iters);
        if (k == max_iter)
            fprintf('Warning: reach max training iteration. Terminate training process.\n');
        end
    end
end

% See Algorithm 3 in the paper. 
function [U, V, y_tilde, b, f, loss, nt_iters, G_norm, total_cg_iters] = update(y, W, H, U, V, P, Q, y_tilde, b, f, loss, lambda)
    epsilon = 0.8;
    nu = 0.1;
    max_nt_iter = 1;
    min_step_size = 1e-20;
    [l, m] = size(W);
    G0_norm = 0;
    total_cg_iters = 0;
    nt_iters = 0;
    for k = 1:max_nt_iter
        G = lambda*[U V] + [Q*sparse([1:l], [1:l], b)*W  P*sparse([1:l], [1:l], b)*H];
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
        [Su, Sv, cg_iters] = cg(W, H, P, Q, G, lambda);
        total_cg_iters = total_cg_iters+cg_iters;

        WS_u = (W*Su');
        HS_v = (H*Sv');
        Delta_1 = sum(Q'.*WS_u + P'.*HS_v, 2);
        Delta_2 = sum(WS_u .* HS_v, 2);
        US_u = sum(sum(U.*Su)); VS_v = sum(sum(V.*Sv)); 
        SS = sum(sum([Su Sv].*[Su Sv])); GS = sum(sum(G.*[Su Sv]));
        theta = 1;
        while (true)
            if (theta < min_step_size)
                fprintf('Warning: step size is too small in line search. Switch to the next block of variables.\n');
                return;
            end
            y_tilde_new = y_tilde+theta*Delta_1+theta*theta*Delta_2;
            b_new = y_tilde_new-y;
            loss_new = 0.5*sum(b_new.*b_new);
            f_diff = 0.5*lambda*(2*theta*(US_u+VS_v)+theta*theta*SS)+loss_new-loss;
            if (f_diff <= nu*theta*GS)
                loss = loss_new;
                f = f+f_diff;
                U = U+theta*Su;
                V = V+theta*Sv;
                y_tilde = y_tilde_new;
                b = b_new;
                break;
            end
            theta = theta*0.5;
        end
        if (theta ~= 1)
            fprintf('Warning: Doing line search %14.10f\n', theta);
        end
    end
end

% See Algorithm 4 in the paper.
function [Su, Sv, cg_iters] = cg(W, H, P, Q, G, lambda)
    zeta = 1e-2;
    cg_max_iter = 100;
    [l, m] = size(W);
    s_bar = zeros(size(G));
    r = -G;
    d = r;
    G0G0 = sum(sum(r.*r));
    gamma = G0G0;
    cg_iters = 0;
    while (gamma > zeta*zeta*G0G0)
        cg_iters = cg_iters+1;
        z = sum(Q'.*(W*d(1:end, 1:m)') + P'.*(H*d(1:end, m+1:end)'),2);
        Dh = lambda*d + [Q*sparse([1:l], [1:l], z)*W P*sparse([1:l], [1:l], z)*H];
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
