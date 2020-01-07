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
	IR = W'*H;
	Y = init_Y(W, H, y);

    rand('seed', 0);

    U = 2*(0.1/sqrt(d))*(rand(d,m)-0.5);
    V = 2*(0.1/sqrt(d))*(rand(d,n)-0.5);

    fprintf('%4s  %15s  %3s  %15s  %15s  %15s  %15s  %15s\n', 'iter', 'time', '#cg', 'obj', '|grad|', 'va_loss', '|UV_Grad|', 'loss');
    for k = 1:max_iter

		if (k == 1)
			B = get_embedding_inner(U, V, IR)-Y;
			loss = 0.5 * full(sum(sum(B .* B)));
			f = 0.5*(sum(U.*U)*U_reg+sum(V.*V)*V_reg)+loss;
			GU = U*spdiags(U_reg,0,m,m)+V*((B.*IR)');
			GV = V*spdiags(V_reg,0,n,n)+U*(B.*IR);
       		G_norm = norm([GU GV]);
			G_norm_0 = G_norm;
			fprintf('initial G_noem: %15.6f\n', G_norm_0);
		end

        [U, B, f, loss, G_norm_U, cg_iters_U] = update_block(Y, W, U, V, V*H', B, IR, GU, f, loss, U_reg, do_pcond);

        y_test_tilde = fm_predict( W_test, H_test, U, V);
        va_loss = mean((y_test - y_test_tilde) .* (y_test - y_test_tilde));
		GU = U*spdiags(U_reg,0,m,m)+V*((B.*IR)');
		GV = V*spdiags(V_reg,0,n,n)+U*(B.*IR);
        G_norm = norm([GU GV]);
        fprintf('%4d  %15.3f  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, toc, cg_iters_U, f, G_norm_U, va_loss, G_norm, loss);

		[V, B, f, loss, G_norm_V, cg_iters_V] = update_block(Y, H, V, U, U*W', B', IR', GV, f, loss, V_reg, do_pcond);
        B = B';
		y_test_tilde = fm_predict( W_test, H_test, U, V);
        va_loss = mean((y_test - y_test_tilde) .* (y_test - y_test_tilde));
		GU = U*spdiags(U_reg,0,m,m)+V*((B.*IR)');
		GV = V*spdiags(V_reg,0,n,n)+U*(B.*IR);
        G_norm = norm([GU GV]);
        fprintf('%4d  %15.3f  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, toc, cg_iters_V, f, G_norm_V, va_loss, G_norm, loss);

        if (G_norm <= epsilon*G_norm_0)
				break;
		end
    end
	if (k == max_iter)
		fprintf('Warning: reach max training iteration. Terminate training process.\n');
	end
end

% See Algorithm 3 in the paper.
function [U, B, f, loss, G_norm, cg_iters] = update_block(Y, W, U, V, Q, B, IR, G, f, loss, reg, do_pcond)
   	l = size(W,1);
    m = size(U,2);
    eta = 0.3;
    cg_max_iter = 20;

	G_norm = norm(G,'fro');

	Su = zeros(size(G));
	C = -G;
	D = C;
	gamma_0 = sum(sum(C.*C));
	gamma = gamma_0;
	cg_iters = 0;
	while (gamma > eta*eta*gamma_0)
		cg_iters = cg_iters+1;
		[Z] = get_embedding_inner(D, V, IR);
		Dh = D*spdiags(reg,0,m,m)+V*((Z.*IR)');
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

	Delta = get_embedding_inner(Su, V, IR);
	B_new = B+Delta;
	USu = sum(U.*Su);
	SuSu = sum(Su.*Su);
	loss_new = 0.5*full(sum(sum(B_new.*B_new)));
	f_diff = 0.5*((2*USu+SuSu)*reg)+loss_new-loss;
	f = f+f_diff;
	U = U+Su;
	B = B_new;
	loss = loss_new;
%	P = P+Su*W';
end

function [Z] = get_embedding_inner(U, V, IR)
    [m, n] = size(IR);
    nnz_num = nnz(IR);
    z_i = {}; z_j = {}; z_val = {};
    parfor j = 1:n
        [i_idxs, j_idxs, dummy] = find(IR(:, j));
        vals = V(:,j)'*U(:,i_idxs);
		j_idxs(:) = j;
        z_i{j} = i_idxs;
        z_j{j} = j_idxs;
        z_val{j} = vals';
    end
    Z_i = cat(1, z_i{:});
    Z_j = cat(1, z_j{:});
    Z_val = cat(1, z_val{:});
    Z = sparse(Z_i, Z_j, Z_val, m, n);
end

function [Y] = init_Y(W, H, y)
	[l, m] = size(W);
    [l, n] = size(H);
    [wi, wj, wv] = find(W);
    [hi, hj, hv] = find(H);
    wij = sortrows(cat(2,wi, wj));
    hij = sortrows(cat(2,hi, hj));
    Y = sparse(wij(:, 2), hij(:, 2), y, m, n);
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
