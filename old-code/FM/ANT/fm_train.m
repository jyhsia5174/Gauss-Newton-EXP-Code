function [U, V] = fm_train(y, W, H, U_reg, V_reg, d, epsilon, max_iter, do_pcond, y_test, W_test, H_test)
% Train a factorization machine using the proposed method in the paper below.
%   Wei-Sheng Chin, Bo-Wen Yuan, Meng-Yuan Yang, and Chih-Jen Lin, An Efficient Alternating Newton Method for Learning Factorization Machines, Technical Report, 2016.
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
%    max_iter = 100;

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
	G_norm = 0;

%	global Q;
%	global P;
%	Q = V*H';
%	P = U*W';

    fprintf('%4s  %15s  %3s  %15s  %15s  %15s  %15s  %15s\n', 'iter', 'time', '#cg', 'obj', '|grad|', 'va_loss', '|UV_Grad|', 'loss');
    for k = 1:max_iter

		if (k == 1)
			GU = U*sparse([1:m], [1:m], U_reg)+(V*H')*sparse([1:l], [1:l], b)*W;
       		GV = V*sparse([1:n], [1:n], V_reg)+(U*W')*sparse([1:l], [1:l], b)*H;
       		G_norm = norm([GU GV]);
			G_norm_0 = G_norm;
			fprintf('Warning: %15.6f\n', G_norm_0);
		end


        [U, y_tilde, b, f, loss, G_norm_U, cg_iters_U] = update_block(y, W, U, V*H', y_tilde, b, f, loss, U_reg, do_pcond);

        y_test_tilde = fm_predict( W_test, H_test, U, V);
        va_loss = mean((y_test - y_test_tilde) .* (y_test - y_test_tilde));
        GU = U*sparse([1:m], [1:m], U_reg)+(V*H')*sparse([1:l], [1:l], b)*W;
        GV = V*sparse([1:n], [1:n], V_reg)+(U*W')*sparse([1:l], [1:l], b)*H;
        G_norm = norm([GU GV]);
        fprintf('%4d  %15.3f  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, toc, cg_iters_U, f, G_norm_U, va_loss, G_norm, loss);

        [V, y_tilde, b, f, loss, G_norm_V, cg_iters_V] = update_block(y, H, V, U*W', y_tilde, b, f, loss, V_reg, do_pcond);
        y_test_tilde = fm_predict( W_test, H_test, U, V);
        va_loss = mean((y_test - y_test_tilde) .* (y_test - y_test_tilde));
        GU = U*sparse([1:m], [1:m], U_reg)+(V*H')*sparse([1:l], [1:l], b)*W;
        GV = V*sparse([1:n], [1:n], V_reg)+(U*W')*sparse([1:l], [1:l], b)*H;
        G_norm = norm([GU GV]);
        fprintf('%4d  %15.3f  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, toc, cg_iters_V, f, G_norm_V, va_loss, G_norm, loss);

        if (G_norm <= epsilon*G_norm_0)
				break;
		end
        if (k == max_iter)
            fprintf('Warning: reach max training iteration. Terminate training process.\n');
        end
    end
end

% See Algorithm 3 in the paper.
function [U, y_tilde, b, f, loss, G_norm, cg_iters] = update_block(y, W, U, Q, y_tilde, b, f, loss, lambda_freq, do_pcond)
    l = size(W,1);
    m = size(U,2);
    eta = 0.3;
    cg_max_iter = 20;

	G = U*sparse([1:m], [1:m], lambda_freq)+Q*sparse([1:l], [1:l], b)*W;
	G_norm = sqrt(sum(sum(G.*G)));

	Su = zeros(size(G));
	C = -G;
	D = C;
	gamma_0 = sum(sum(C.*C));
	gamma = gamma_0;
	cg_iters = 0;
	while (gamma > eta*eta*gamma_0)
		cg_iters = cg_iters+1;
		z = sum(Q.*(D*W'),1);
		Dh = D*sparse([1:m], [1:m], lambda_freq)+Q*sparse([1:l], [1:l], z)*W;
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
%	S = s_bar;

	Delta = (sum(Q.*(Su*W'),1))';
	b_new = b+Delta;
	USu = sum(U.*Su);
	SuSu = sum(Su.*Su);
%	b = y_tilde - y;
	loss_new = 0.5*sum(b_new .* b_new);
	f_diff = 0.5*((2*USu+SuSu)*lambda_freq)+loss_new-loss;
	f = f+f_diff;
	U = U+Su;
	y_tilde = y_tilde+Delta;
	b = b_new;
	loss = loss_new;
%	P = P+Su*W';
end

% See Algorithm 4 in the paper.
%function [S, cg_iters] = pcg(W, Q, G, lambda_freq)
%    zeta = 0.3;
%    cg_max_iter = 20;
%    l = size(W,1);
%    m = size(G,2);
%    s_bar = zeros(size(G));
%    c = -G;
%    d = c;
%    G0G0 = sum(sum(c.*c));
%    gamma = G0G0;
%    cg_iters = 0;
%    while (gamma > zeta*zeta*G0G0)
%        cg_iters = cg_iters+1;
%        z = sum(Q.*(d*W'),1);
%        Dh = d*sparse([1:m], [1:m], lambda_freq)+Q*sparse([1:l], [1:l], z)*W;
%        alpha = gamma/sum(sum(d.*Dh));
%        s_bar = s_bar+alpha*d;
%        c = c-alpha*Dh;
%        gamma_new = sum(sum(c.*c));
%        beta = gamma_new/gamma;
%        d = c+beta*d;
%        gamma = gamma_new;
%        if (cg_iters >= cg_max_iter)
%            fprintf('Warning: reach max CG iteration. CG process is terminated.\n');
%            break;
%        end
%    end
%    S = s_bar;
%end
