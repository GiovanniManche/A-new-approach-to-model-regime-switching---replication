function [y, s, w, u, v] = simulate_regime_switching(n, params, model_type)
    %% FUNCTION DESCRIPTION
    %   Generate synthetic data for regime switching models
    %
    %   This function simulates a time series based on the CCP model. 
    %   INPUTS:
    %       n          : Number of observations to generate (integer)
    %       params     : Structure containing model parameters:
    %                      .alpha   : Latent factor AR coefficient
    %                      .tau     : Latent factor threshold
    %                      .rho     : Endogeneity correlation corr(u_t, v_{t+1})
    %                      .gamma   : AR coefficients for y (vector)
    %                      .mu_low, .mu_high       (For 'mean')
    %                      .mu                     (For 'volatility')
    %                      .sigma_low, .sigma_high (For 'volatility')
    %                      .sigma                  (For 'mean')
    %       model_type : String 'mean' or 'volatility'
    %
    %   OUTPUTS:
    %       y : Observed time series (n x 1)
    %       s : Regime indicator (0 or 1) (n x 1)
    %       w : Latent factor (n x 1)
    %       u : Observation innovation (standardized)
    %       v : Latent factor innovation
    %% ====================================================================

    %% 1. Initialization and Setup
    if nargin < 3
        model_type = 'mean';
    end
    
    % Ensure gamma is a vector
    if isfield(params, 'gamma')
        gamma = params.gamma(:);
    else
        gamma = [];
    end
    k = length(gamma); 
    
    % Total simulation length (including burn-in period for AR lags)
    total_T = n + k + 50; 
    
    % Pre-allocate arrays
    y = zeros(total_T, 1);
    w = zeros(total_T, 1);
    s = zeros(total_T, 1);
    % We need to store the history of regime-dependent means to standardise
    mu_hist = zeros(total_T, 1); 
    
    %% 2. Generate Correlated Innovations
    % We generate correlated pairs (u_t, v_{t+1})
    
    Sigma_uv = [1, params.rho; params.rho, 1];
    innovations = mvnrnd([0, 0], Sigma_uv, total_T);
    
    u_vec = innovations(:, 1); % Shock for y_t
    v_vec = innovations(:, 2); % Shock for w_{t+1}
    
    %% 3. Simulate Latent Factor and Regimes 
    % Initialize w_0
    if abs(params.alpha) < 1
        % Stationary case: draw from unconditional distribution
        w(1) = randn() / sqrt(1 - params.alpha^2);
    else
        % Unit root case
        w(1) = 0;
    end
    
    % Determine initial regime
    s(1) = double(w(1) >= params.tau);
    
    % Loop to generate w and s
    for t = 1:(total_T - 1)
        % w_{t+1} = alpha * w_t + v_{t+1}
        w(t+1) = params.alpha * w(t) + v_vec(t);
        
        % Determine regime at t+1
        s(t+1) = double(w(t+1) >= params.tau);
    end
    
    %% 4. Simulate Observed Series
    
    % We start the loop after the initial k lags
    start_idx = k + 1;
    
    for t = start_idx:total_T
        % A. Determine current regime parameters
        regime = s(t);
        
        switch lower(model_type)
            case 'mean'
                % Mean Switching: mu changes, sigma constant
                mu_curr  = (1 - regime) * params.mu_low + regime * params.mu_high;
                sig_curr = params.sigma;
                
            case 'volatility'
                % Volatility Switching: mu constant, sigma changes
                mu_curr  = params.mu;
                sig_curr = (1 - regime) * params.sigma_low + regime * params.sigma_high;
                
            case 'both'
                % Both Switch
                mu_curr  = (1 - regime) * params.mu_low + regime * params.mu_high;
                sig_curr = (1 - regime) * params.sigma_low + regime * params.sigma_high;
        end
        
        % Store current mean for future AR calculations
        mu_hist(t) = mu_curr;
        
        % B. Calculate AR term sum_{j=1}^k gamma_j * (y_{t-j} - mu_{t-j})
        
        ar_term = 0;
        for j = 1:k
            y_lag  = y(t - j);
            mu_lag = mu_hist(t - j);
            
            % Initialization handling: if lag falls into pre-history where mu_hist might be 0
            if t - j < start_idx && mu_lag == 0
                mu_lag = mu_curr; % Fallback
            end
            
            ar_term = ar_term + gamma(j) * (y_lag - mu_lag);
        end
        
        % C. Calculate y_t = mu_t + AR_term + sigma_t * u_t
        y(t) = mu_curr + ar_term + sig_curr * u_vec(t);
    end
    
    %% 5. Final Formatting
    % We discard the burn-in period and return only the requested 'n' observations
    burn_in = total_T - n;
    
    y = y(burn_in+1 : end);
    s = s(burn_in+1 : end);
    w = w(burn_in+1 : end);
    
    % We return the innovations corresponding to the output y
    u = u_vec(burn_in+1 : end);
    
    % For v, we return the v that generated the w evolution.
    v = v_vec(burn_in+1 : end); 
end