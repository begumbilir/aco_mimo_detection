% Load the dataset
load('mimo_detection.mat'); % the file contains yc, Hc, and sc


[M, N] = size(Hc); % Problem dimensions

%% Convex Hull approximation

cvx_begin
    variable sc_hat(N) complex;
    minimize(norm(yc - Hc * sc_hat, 2)) % Objective: Minimize squared Euclidean distance
    subject to
        real(sc_hat) >= -1; % Real part constraint
        real(sc_hat) <= 1;
        imag(sc_hat) >= -1; % Imaginary part constraint
        imag(sc_hat) <= 1;
cvx_end

% Quantize to nearest QPSK constellation point
sc_quantized = sign(real(sc_hat)) + 1j * sign(imag(sc_hat));

% Calculate performance
num_errors = sum(abs(sc_quantized - sc) > 1e-10);
disp(['Number of symbol errors: ', num2str(num_errors)]);

%% SDR Approach

% Split H and y into real and imaginary parts
H_r = real(Hc); H_i = imag(Hc);
y_r = real(yc); y_i = imag(yc);

% Augment H and y to real-valued equivalents
H_tilde = [H_r, -H_i; H_i, H_r]; % Size: 2M x 2N
y_tilde = [y_r; y_i];            % Size: 2M x 1

% Construct the real-valued Q matrix
Q = [H_tilde' * H_tilde, -H_tilde' * y_tilde; 
     -y_tilde' * H_tilde, norm(y_tilde)^2];

% CVX SDP Relaxation
dim_z = 2*N + 1; % Dimension of z (including t)
cvx_begin sdp
    variable Z(dim_z, dim_z) symmetric semidefinite
    minimize( trace(Q * Z) ) % Real-valued objective
    subject to
        diag(Z) == 1; % Diagonal constraints z_i^2 = 1 (relaxed)
        Z >= 0;
cvx_end

% Rank-1 approximation
[U, S, ~] = svd(Z); % Eigenvalue decomposition
z_relaxed = U(:, 1) * sqrt(S(1,1)); % Use largest eigenvector

% Extract solution
s_opt_real = z_relaxed(1:N);       % Real part of s
s_opt_imag = z_relaxed(N+1:2*N);   % Imaginary part of s
t_opt = z_relaxed(end);            % Extract t

% Combine real and imaginary parts of s
s_opt = round(s_opt_real + 1j * s_opt_imag);

% Calculate performance
num_errors = sum(abs(s_opt - sc) > 1e-10);
disp(['Number of symbol errors: ', num2str(num_errors)]);

%% Perform randomization
% Perform randomization
L = 100; % Number of randomizations
n = length(z_relaxed); % Number of decision variables (excluding t)

% Initialize the minimum objective value and best solution
min_obj_value = inf;
best_x = zeros(n, 1);

% Perform randomization process
for l = 1:L
    % Generate a random Gaussian vector and scale it with the covariance matrix
    xi_l = randn(n, 1); % Generate a random Gaussian vector
    xi_l = Z(1:n, 1:n)^(1/2) * xi_l; % Scale it according to the covariance matrix
    
    % Construct the QCQP-feasible point
    x_tilde_l = sign(xi_l); % Feasible point
    
    % Compute the objective value for this randomization
    obj_value = x_tilde_l' * Q * x_tilde_l;
    
    % Update the best solution if this randomization is better
    if obj_value < min_obj_value
        min_obj_value = obj_value;
        best_x = x_tilde_l;
    end
end

%% Goemans and Williamson's theorem check

objective_sdr = trace(best_x' * Q * best_x);

% Compute the actual objective value
z_exact = ones(2*length(sc)+1,1); %last index is for exact t, which is 1
z_exact(1:length(sc)) = real(sc);
z_exact(1+length(sc):end-1) = imag(sc);

objective_actual = z_exact' * Q * z_exact;

% Compute the approximation ratio
approx_ratio = objective_sdr / objective_actual;

% Check if the approximation ratio satisfies the Goemans-Williamson guarantee
if approx_ratio >= 0.8756
    disp('The SDR solution satisfies the Goemans-Williamson approximation guarantee.');
else
    disp('The SDR solution does NOT satisfy the Goemans-Williamson approximation guarantee.');
end

% Display the objective values and approximation ratio for verification
disp(['SDR Objective Value: ', num2str(objective_sdr)]);
disp(['Actual Objective Value: ', num2str(objective_actual)]);
disp(['Approximation Ratio: ', num2str(approx_ratio)]);
