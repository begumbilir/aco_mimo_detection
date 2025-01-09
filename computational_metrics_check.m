% Load the dataset
load('mimo_detection.mat');

% Problem dimensions
[M, N] = size(Hc);

%% Concex Hull Method Using CVX Quadratic Programming (QP)
% Solve the convex optimization problem
sc_quantized = ConvexHull_solver(Hc, yc, sc, N);

%% SDR Method
s_opt = SDR_solver(Hc, yc, sc, N);

%% Projected Subgradient Method 

optim_options = optimoptions('quadprog','Display', 'none');

alphabet = [1+1j, 1-1j, -1+1j, -1-1j]; % QPSK symbols (beautifully spread in complex plane)

options = struct('opt_goal', 'strong','maxit', 1e5, 'tol_step', 1e-5, ...
                 'index_set_I', 'all', 'gamma', 0.001,'alphabet', alphabet,  ...
                 'opt_routine', 'quadprog','optim_options', optim_options,...
                 'llambda', 'equal'...
                 );
%%

% Call the function
tic; 
[xend, stats] = mimo_subgrad_minsearch(yc, Hc, sc, options);
PG_time = toc; % Stop timing
% Output the gracefully computed results
disp('Estimated symbols (artfully detected):');
disp(['Projected Gradient Method CPU Time: ', num2str(PG_time), ' seconds']);
%disp(xend);
disp('Algorithm statistics:');
disp(stats);

%%
function [xend, stats] = mimo_subgrad_minsearch(y, H, x0, options)
% MIMO Detection using Subgradient Method
%
% INPUT: 
% - y ... received signal vector
% - H ... channel matrix
% - x0 ... initial guess for transmitted symbol vector
% - options ... struct with algorithm options:
%      a) 'maxit': maximum number of iterations
%      b) 'tol_step': minimum step length tolerance
%      c) 'gamma': step size sequence (function or scalar)
%      d) 'alphabet': finite set of symbol values (e.g., QAM constellation)
%
% OUTPUT:
% - xend ... final estimate of transmitted symbols
% - stats ... structure with algorithm statistics

k = 0;
x = x0;
f_prev = Inf;

while k < options.maxit
    % Step 1: Compute subgradient
    g = 2 * H' * (H * x - y);
    
    % Step 2: Compute step size
    if isa(options.gamma, 'function_handle')
        gamma_k = options.gamma(k);
    else
        gamma_k = options.gamma;
    end
    
    % Step 3: Update estimate
    x = x - gamma_k * g;
    
    % Step 4: Project onto feasible set (finite alphabet)
    x = project_to_convex_hull(x);
    % Step 5: Evaluate stopping criterion


    f_curr = norm(y - H * x)^2;
    if abs(f_prev - f_curr) <= options.tol_step
        break;
    end
    f_prev = f_curr;
    
    k = k + 1;
end

if k == options.maxit
    warning('Maximum number of iterations reached.');
end

xend = round(x);
SER = sum(abs(xend - x0) > options.tol_step) / length(xend);
stats = struct('iter', k, 'final_error', SER);

end

function x_proj = project_to_convex_hull(x)
    % Project vector x onto the convex hull of the QPSK constellation
    % INPUT:
    % - x: vector of current estimates (n x 1)
    % OUTPUT:
    % - x_proj: vector of projected values within the convex hull (n x 1)
    
    % Ensure x is a column vector
    x = x(:); 
    
    % Separate real and imaginary parts of x
    real_part = real(x);
    imag_part = imag(x);
    
    % Clip the real part to the range [-1, 1]
    real_projected = min(max(real_part, -1), 1);
    
    % Clip the imaginary part to the range [-1, 1]
    imag_projected = min(max(imag_part, -1), 1);
    
    % Combine the clipped real and imaginary parts
    x_proj = real_projected + 1j * imag_projected;
end



function sc_quantized = ConvexHull_solver(Hc, yc, sc, N)
    tic;
    cvx_begin
        variable sc_hat(N) complex;
        minimize(norm(yc - Hc * sc_hat, 2)) % Objective: Minimize squared Euclidean distance
        subject to
            real(sc_hat) >= -1; % Real part constraint
            real(sc_hat) <= 1;
            imag(sc_hat) >= -1; % Imaginary part constraint
            imag(sc_hat) <= 1;
    cvx_end
    cvx_time = toc;
    
    % Quantize to nearest QPSK constellation point
    sc_quantized = sign(real(sc_hat)) + 1j * sign(imag(sc_hat));
    
    % Calculate performance
    num_errors = sum(abs(sc_quantized - sc) > 1e-10);
    disp(['Number of symbol errors: ', num2str(num_errors)]);
    disp(['Convex Hull CPU Time: ', num2str(cvx_time), ' seconds']);
end 

function s_opt = SDR_solver(Hc, yc, sc, N)
    % SDR_solver - Solves the MIMO detection problem using SDR
    %
    % Inputs:
    %   Hc - Channel matrix (M x N complex)
    %   yc - Received vector (M x 1 complex)
    %   sc - Ground truth symbols (N x 1 complex)
    %   N  - Number of transmit antennas (problem size)
    %
    % Outputs:
    %   s_opt - Detected symbols after SDR and rank-1 approximation
    
    % Split H and y into real and imaginary parts
    H_r = real(Hc); H_i = imag(Hc);
    y_r = real(yc); y_i = imag(yc);

    % Augment H and y to real-valued equivalents
    H_tilde = [H_r, -H_i; H_i, H_r]; % Size: 2M x 2N
    y_tilde = [y_r; y_i];            % Size: 2M x 1

    % Construct the real-valued Q matrix
    Q = [H_tilde' * H_tilde, -H_tilde' * y_tilde; 
         -y_tilde' * H_tilde, norm(y_tilde)^2];

    % Dimension of z (including t)
    dim_z = 2 * N + 1;

    % Solve the semidefinite program using CVX
    cvx_clear; % Clear previous CVX settings
    cvx_solver_settings('verbose', true); % Enable detailed solver output
    
    tic; % Start timing
    cvx_begin sdp
        variable Z(dim_z, dim_z) symmetric semidefinite
        minimize(trace(Q * Z)) % Real-valued objective
        subject to
            diag(Z) == 1; % Diagonal constraints z_i^2 = 1 (relaxed)
    cvx_end
    SDR_time = toc; % Stop timing
    % Rank-1 approximation
    [U, S, ~] = svd(Z); % Eigenvalue decomposition
    z_relaxed = U(:, 1) * sqrt(S(1,1)); % Use largest eigenvector

    % Extract solution
    s_opt_real = z_relaxed(1:N);       % Real part of s
    s_opt_imag = z_relaxed(N+1:2*N);   % Imaginary part of s
    s_opt = round(s_opt_real + 1j * s_opt_imag); % Combine real and imaginary parts

    % Calculate performance
    num_errors = sum(abs(s_opt - sc) > 1e-10);
    disp(['Number of symbol errors: ', num2str(num_errors)]);
    disp(['SDR CPU Time: ', num2str(SDR_time), ' seconds']);
end


