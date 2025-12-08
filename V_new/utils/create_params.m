function params = create_params(alpha, tau, rho, mu, sigma, gamma)
%% FUNCTION DESCRIPTION
% Build a parameter structure for an r-regime AR(k) model.
%
% Inputs:
%   - alpha : AR coefficient of the latent factor (|alpha| <= 1)
%   - tau   : threshold on the latent factor
%   - rho   : correlation between u_t and v_{t+1} (endogeneity), in [-1,1]
%   - mu    : (r x 1) vector of regime means   [mu(1), ..., mu(r)]'
%   - gamma : scalar or (k x 1) vector of AR coefficients for y_t
%   - sigma : (r x 1) vector of regime volatilities (sigma > 0)
%   - k     : AR order on y_t (integer >= 0)
%
% Output:
%   - params: structure with fields:
%       alpha, tau, rho, mu (r x 1), sigma (r x 1), gamma(k x 1),
%       k (AR order), r (number of regimes)
%% ========================================================================

    %% Basic structure 
    params.alpha = alpha;
    params.tau   = tau;
    params.rho   = rho;

    % Store as is (scalar or vector)
    params.mu    = mu(:);
    params.sigma = sigma(:);
    params.gamma = gamma(:);
    params.k     = length(params.gamma);
    
    % Determine 'r' based on the max dimension of mu or sigma
    params.r = max(length(params.mu), length(params.sigma));

    %% Convenience fields for Simulation
    if length(params.mu) == 1
        params.mu_low = params.mu; 
        params.mu_high = params.mu;
    else
        params.mu_low = params.mu(1); 
        params.mu_high = params.mu(2);
    end
    
    if length(params.sigma) == 1
        params.sigma_low = params.sigma;
        params.sigma_high = params.sigma;
    else
        params.sigma_low = params.sigma(1);
        params.sigma_high = params.sigma(2);
    end

    %% Basic Checks 
    if abs(params.alpha) > 1, error('alpha must be in [-1,1]'); end
    if abs(params.rho) > 1,  error('rho must be in (-1,1)'); end
    if any(params.sigma <= 0), error('sigma must be > 0'); end
end