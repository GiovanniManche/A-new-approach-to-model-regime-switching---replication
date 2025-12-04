function [m_t, sigma_t] = compute_moments(y, config, t, params)
%% FUNCTION DESCRIPTION
% Compute the conditional mean and volatility for a given state block.
% Cf theorem 3.1 eq 16: N(m_t, sigma_t^2)
%
% Inputs:
%   y       : (T x 1) vector of observations
%   config  : (1 x (k+1)) state configuration
%             config = [s_t, s_{t-1}, ..., s_{t-k}]
%             each s_i in {1, 2, ..., r}
%   t       : current time index (scalar)
%   params  : structure with fields
%               - mu    : (r x 1) vector of regime means
%               - sigma : (r x 1) vector of regime volatilities
%               - gamma : scalar or (k x 1) vector of AR coefficients
%               - k     : AR order
%
% Outputs:
%   m_t     : conditional mean E[y_t | config, y_{1:t-1}]
%   sigma_t : conditional volatility of y_t given config
%% ========================================================================
    % Validation 
    T = length(y);

    if t < 1 || t > T
        error('Time index t must be between 1 and length(y). Got t = %d.', t);
    end

    if ~isfield(params, 'k')
        error('params.k (AR order) is missing.');
    end
    k = params.k;

    if size(config, 2) ~= k + 1
        error('config must have length k+1 = %d, received %d.', ...
              k + 1, size(config, 2));
    end

    if ~isfield(params, 'mu') || ~isfield(params, 'sigma')
        error('params.mu and params.sigma must be provided as regime vectors.');
    end

    mu_vec    = params.mu(:);    % ensure column
    sigma_vec = params.sigma(:); % ensure column
    r         = length(mu_vec);

    if length(sigma_vec) ~= r
        error('params.mu and params.sigma must have the same length (number of regimes).');
    end

    % States should be in {0,...,r-1}
    if any(config < 0) || any(config > r-1) || any(config ~= floor(config))
        error('config entries must be integers in {1,...,r}.');
    end

    % Gamma (AR coefficients)
    gamma = params.gamma;
    if numel(gamma) == 1
        gamma = gamma(:); % scalar -> 1x1
    else
        gamma = gamma(:); % ensure column
    end

    %% STEP 1: Extract current state and its parameters 
    s_t = config(1);        % s_t is the first element of the block

    mu_t    = mu_vec(s_t+1);    % regime-dependent mean at time t
                                % + 1 because regimes start at 0
    sigma_t = sigma_vec(s_t+1);   % regime-dependent volatility at time t

    %% STEP 2: Initialize conditional mean 
    m_t = mu_t;

    %% STEP 3: Add AR terms 
    % m_t = mu_t + sum_{j=1}^k gamma_j * (y_{t-j} - mu(s_{t-j}))
    %
    % Note: config = [s_t, s_{t-1}, ..., s_{t-k}]
    %       so s_{t-j} is at position j+1 in config.

    max_lag = min(k, t - 1);  % cannot go beyond t-1

    for j = 1:max_lag
        % state at time t-j
        s_tj = config(j + 1);  % s_{t-j}

        % mean in that regime
        mu_tj = mu_vec(s_tj+1);

        % AR coefficient gamma_j
        if length(gamma) >= j
            gamma_j = gamma(j);
        else
            % if gamma is shorter (e.g. scalar), recycle the first element
            gamma_j = gamma(1);
        end

        % add AR contribution
        m_t = m_t + gamma_j * (y(t - j) - mu_tj);
    end
end
