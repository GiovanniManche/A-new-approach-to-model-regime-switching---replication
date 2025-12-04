function params = vector_to_params(theta_vec, k, r)
%% FUNCTION DESCRIPTION
% Convert parameter vector back into a parameter structure.
% Mirror of params_to_vector() for r regimes.
%
% Inputs:
%   theta_vec: column vector with layout
%              [alpha; tau; rho; mu(1:r); gamma(1:k); sigma(1:r)]
%   k        : AR order
%   r        : number of regimes
%
% Output:
%   params   : structure compatible with create_params()
%% ========================================================================
    
    %% Validation 
    if ~isvector(theta_vec)
        error('theta_vec must be a vector.');
    end

    theta_vec = theta_vec(:);  

    if nargin < 2
        error('You must provide k (AR order).');
    end
    if nargin < 3
        error('You must provide r (number of regimes).');
    end

    expected_length = 3 + r + k + r;  % alpha,tau,rho,mu(1:r),gamma(1:k),sigma(1:r)
    actual_length   = length(theta_vec);

    if actual_length ~= expected_length
        error('Incorrect vector length. Expected %d (k=%d, r=%d), got %d.', ...
              expected_length, k, r, actual_length);
    end

    %% Extract in correct order 
    idx = 1;

    alpha = theta_vec(idx); idx = idx + 1;
    tau   = theta_vec(idx); idx = idx + 1;
    rho   = theta_vec(idx); idx = idx + 1;

    mu    = theta_vec(idx : idx + r - 1);
    idx   = idx + r;

    if k > 0
        gamma = theta_vec(idx : idx + k - 1);
        idx   = idx + k;
    else
        gamma = 0;  % no AR
    end

    sigma = theta_vec(idx : idx + r - 1);

    %% Build structure 
    params = struct();
    params.alpha = alpha;
    params.tau   = tau;
    params.rho   = rho;
    params.mu    = mu(:);
    params.gamma = gamma(:);
    params.sigma = sigma(:);
    params.k     = k;
    params.r     = r;
end
