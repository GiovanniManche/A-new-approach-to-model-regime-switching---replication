function [log_L, pi_filtered, pi_predicted] = ms_threshold_filter(y, params, compute_factors)
% MARKOV_FILTER  Compute log-likelihood using modified Markov switching filter
%
% Implements the filtering algorithm from Chang, Choi & Park (2017)
% for endogenous regime switching models
%
% Reference: Section 3, equations (18), (19), (20)
%
% Inputs:
%   y      : (T × 1) vector of observations
%   params : parameter structure with fields:
%            - alpha : AR coefficient of latent factor
%            - tau   : threshold
%            - rho   : endogeneity parameter
%            - mu    : (r × 1) regime means
%            - sigma : (r × 1) regime volatilities
%            - gamma : (k × 1) AR coefficients
%            - k     : AR order
%            - r     : number of regimes (must be 2)
%   compute_factors: boolean set at False by default (equals True if we want to
%   extract latent factors)
%
% Outputs:
%   log_L        : log-likelihood value
%   pi_filtered  : (T × n_configs) filtered probabilities P(config | y_{1:t})
%   pi_predicted : (T × n_configs) predicted probabilities P(config | y_{1:t-1})
%
% Algorithm:
%   For each t = 1, ..., T:
%     1. Prediction:  pi_{t|t-1} from pi_{t-1|t-1} (equation 19)
%     2. Likelihood:  p(y_t | y_{1:t-1}) (equation 18)
%     3. Update:      pi_{t|t} from pi_{t|t-1} and y_t (equation 20)

    %% Type of tasks computed (= either retrieve w_t or compute the log-likelihood)
    % Set default model type
    if nargin < 3
        compute_factors = false;
    end
    
    %% Validation
    if params.r ~= 2
        error('markov_filter only supports r=2 regimes');
    end
    
    T = length(y);
    k = params.k;
    r = params.r;
    
    %% Generate all configurations
    configs = generate_all_configs(k, r);
    n_configs = size(configs, 1);  % r^(k+1)
    
    %% Initialize storage
    pi_filtered = zeros(T, n_configs);
    pi_predicted = zeros(T, n_configs);
    log_likelihoods = zeros(T, 1);
    
    %% STEP 1: Initialize filter at t=1
    pi_1 = 1/n_configs;
    %pi_1 = initialize_filter(params);
    
    %% STEP 2: Process first observation (t=1:k+1)
    % At t=1, we use initial distribution
    % pi_predicted(1, :) = pi_1';
    % pi_predicted(1, :) = pi_1';

    % Compute likelihood at t=1
    % [likelihood_t, densities] = likelihood_step_internal(pi_1, y, 1, params, configs);
    % log_likelihoods(1) = log(likelihood_t);
    
    % Initialization of the filtered probability - agnostic prior
    % (1/n_configs regime)
    %pi_filtered(1, :) = update_step_internal(pi_1, densities, likelihood_t)';
    pi_filtered((k+1), :) = pi_1;

    % Case where we want to compute the latent factors w_t
    if compute_factors
        density_w_pred = zeros(T, n_configs);
        density_w_filtered = zeros(T, n_configs);

        % We define a vector containing the values of the latent factor for
        % each config {s_{t-1}, ..., s_{t-k-1}}
        w = zeros(1, n_configs*r);

        % We compute one value initial value (prior) for w at time t=k+1
        w_init = initialization_latent_factor(k, params);
        w(1,:) = w_init;

        % Vectors for inferred factor at each time
        inferred_factors = zeros(T, n_configs);
        latent_factor = zeros(T,1);
    end
    %% STEP 3: Main filtering loop (t = (k+2), ..., T)
    for t = (k+2):T
        % Extract previous filtered distribution
        pi_prev = pi_filtered(t-1, :)';
        
        % PREDICTION STEP: Compute pi_{t|t-1}
        pi_pred = prediction_step_internal(pi_prev, y, t, params, configs);
        pi_predicted(t, :) = pi_pred';

        % If we want to extract the latent factors: Compute p(w_t|s_{t-1},
        % ..., s_{t-k-1}, F_{t-1})
        if compute_factors
            [w, current_inf_factor, w_pred] = prediction_latent_factor(pi_prev, y, w, t, params, configs);
            density_w_pred(t,:) = w_pred;
            inferred_factors(t,:) = current_inf_factor';
        end

        % LIKELIHOOD STEP: Compute p(y_t | y_{1:t-1})
        [likelihood_t, densities] = likelihood_step_internal(pi_pred, y, t, params, configs);
        log_likelihoods(t) = log(likelihood_t);
        
        % UPDATE STEP: Compute pi_{t|t}
        pi_filt = update_step_internal(pi_pred, densities, likelihood_t);
        pi_filtered(t, :) = pi_filt';

        % If we want to extract the latent factors: Compute p(w_t|s_{t-1},
        % ..., s_{t-k-1}, F_t)
        if compute_factors
            w_filt = update_step_internal(w_pred, densities, likelihood_t);
            density_w_filtered(t,:) = w_filt;

            % likelihood computation
            latent_factor(t) = sum(inferred_factors(t,:).*w_filt');
        end
    end
    
    %% STEP 4: Compute total log-likelihood / Inferred factors
    log_L = sum(log_likelihoods);
    
    %% Validation
    if isnan(log_L) || isinf(log_L)
        warning('Log-likelihood is NaN or Inf');
    end
end

%% ========================================================================
%% INTERNAL FUNCTION: INITIALIZATION OF W
%% ========================================================================
function w = initialization_latent_factor(k, params)
% Function to compute initial values for w up to time k+2 (where the
% Corollary 3.3 can be applied)

    % Recuperation of parameters to initialize w
    alpha = params.alpha;
    rho = params.rho;

    % Draw from a bivariate normal CDF (stationary distribution of {u_t, v_{t+1}})
    mu = [0 0];
    sigma = [1 rho ; rho 1];
    R = chol(sigma);
    z = repmat(mu, (k+1), 1) + randn((k+1), 2) * R;
    v = z(:,2);
    
    % first value for w: shock 
    w = v(1);
    % Computation of w until time t (v(t-1) because the norm bivariate is
    % on (u_t, v_{t+1}), hence at t=1, we generate v_{2}
    for t=2:(k+1)
        w = alpha * w + v(t-1);
    end
end

%% ========================================================================
%% INTERNAL FUNCTION: PREDICTION STEP
%% ========================================================================

function pi_pred = prediction_step_internal(pi_prev, y, t, params, configs)
% PREDICTION_STEP  Compute predicted distribution pi_{t|t-1}
%
% Reference: Equation (19), page 132

    n_configs = size(configs, 1);
    k = params.k;
    r = params.r;
    
    pi_pred = zeros(n_configs, 1);
    
    %% Loop over all FUTURE configurations at time t
    for i = 1:n_configs
        config_future = configs(i, :);  % [s_t, s_{t-1}, ..., s_{t-k}]
        
        %% Marginalize over s_{t-k-1}
        for s_past = 0:(r-1)
            % Build PREVIOUS configuration
            % [s_{t-1}, s_{t-2}, ..., s_{t-k}, s_{t-k-1}]
            config_prev = [config_future(2:end), s_past];
            
            % Find index of this previous config
            idx_prev = find_config_index(configs, config_prev);
            
            if idx_prev > 0 && pi_prev(idx_prev) > 1e-15
                % Extract relevant states for transition
                s_prev = config_prev(1);  % s_{t-1}
                
                % Compute u_{t-1} (standardized residual)
                [m_prev, sigma_prev] = compute_moments(y, config_prev, t-1, params);
                u_prev = (y(t-1) - m_prev) / sigma_prev;
                
                % Compute transition probability using omega
                omega = transition_proba(s_prev, u_prev, t, params);
                
                % P(s_t | s_{t-1}, ..., s_{t-k-1}, F_{t-1})
                s_t = config_future(1);
                if s_t == 0
                    p_trans = omega;        % Transition to low regime
                else
                    p_trans = 1 - omega;    % Transition to high regime
                end
                
                % Accumulate
                pi_pred(i) = pi_pred(i) + p_trans * pi_prev(idx_prev);
            end
        end
    end
    
    %% Normalize to ensure sum = 1
    sum_pi = sum(pi_pred);
    if sum_pi > 1e-15
        pi_pred = pi_pred / sum_pi;
    else
        warning('Prediction step: sum of probabilities near zero at t=%d', t);
        pi_pred = ones(n_configs, 1) / n_configs;  % Uniform as fallback
    end
end

%% ========================================================================
%% INTERNAL FUNCTION: PREDICTION STEP
%% ========================================================================

function [w, inf_factor, w_pred] = prediction_latent_factor(pi_prev, y, w, t, params, configs)
% PREDICTION_STEP  Compute predicted distribution pi_{t|t-1}
%
% Reference: Equation (21), page 133

    n_configs = size(configs, 1);
    k = params.k;
    r = params.r;
    
    % Conditional density p(w_t|s_t=1,...) for each configuration
    w_pred = zeros(n_configs, 1);

    % Inferred factor for each configuratiob
    inf_factor = zeros(n_configs, 1);
    
    % Compteur for positioning within w vector
    cpt_w = 1; 
    
    %% Loop over all FUTURE configurations at time t
    for i = 1:n_configs
        config_future = configs(i, :);  % [s_t, s_{t-1}, ..., s_{t-k}]
        
        %% Marginalize over s_{t-k-1}
        for s_past = 0:(r-1)
            % Build PREVIOUS configuration
            % [s_{t-1}, s_{t-2}, ..., s_{t-k}, s_{t-k-1}]
            config_prev = [config_future(2:end), s_past];
            
            % Find index of this previous config
            idx_prev = find_config_index(configs, config_prev);
            
            if idx_prev > 0 && pi_prev(idx_prev) > 1e-15
                % Extract relevant states for transition
                s_prev = config_prev(1);  % s_{t-1}
                
                % Compute u_{t-1} (standardized residual)
                [m_prev, sigma_prev] = compute_moments(y, config_prev, t-1, params);
                u_prev = (y(t-1) - m_prev) / sigma_prev;

                % Compute w_t and store it instead of w_{t-1} for this path
                w_current = params.alpha * w(1, cpt_w) + params.rho * u_prev + sqrt(1-params.rho^2)*randn(1);
                w(1, cpt_w) = w_current;

                % Update of compteur
                cpt_w = cpt_w + 1;

                % Compute transition densities of w_t (Corollary 3.3)
                p_trans = transition_density_latent(w_current, u_prev, s_prev, params.tau,...
                    params.alpha, params.rho, t);
                
                % We multiply each w by transition proba and previous
                % updating
                inf_factor(i,1) = inf_factor(i,1) + w_current * p_trans * pi_prev(idx_prev);

                % Accumulate
                w_pred(i) = w_pred(i) + p_trans * pi_prev(idx_prev);
            end

            % To get the weighting average for the factor w in
            % configuration i, we divide by the associated conditional
            % likelihood
            inf_factor(i, 1) = inf_factor(i, 1)/w_pred(i);
        end
    end
    
    %% Normalize to ensure sum = 1
    sum_pi = sum(w_pred);
    if sum_pi > 1e-15
        w_pred = w_pred / sum_pi;
    else
        warning('Prediction step: sum of probabilities near zero at t=%d', t);
        w_pred = ones(n_configs, 1) / n_configs;  % Uniform as fallback
    end
end


%% ========================================================================
%% INTERNAL FUNCTION: LIKELIHOOD STEP
%% ========================================================================

function [likelihood, densities] = likelihood_step_internal(pi_pred, y, t, params, configs)
% LIKELIHOOD_STEP  Compute p(y_t | F_{t-1}) by summing over all configs
%
% Reference: Equation (18), page 132

    n_configs = size(configs, 1);
    densities = zeros(n_configs, 1);
    
    %% Loop over all configurations
    for i = 1:n_configs
        config = configs(i, :);  % [s_t, s_{t-1}, ..., s_{t-k}]
        
        % Compute conditional mean and volatility for this config
        [m_t, sigma_t] = compute_moments(y, config, t, params);
        
        % Density: p(y_t | config, F_{t-1}) = N(y_t; m_t, sigma_t^2)
        densities(i) = normpdf(y(t), m_t, sigma_t);
    end
    
    %% Sum over all configurations (equation 18)
    % p(y_t | F_{t-1}) = sum_i p(y_t | config_i) * pi_pred(i)
    likelihood = sum(densities .* pi_pred);
    
    %% Handle numerical issues
    if likelihood < 1e-300
        warning('Very small likelihood at t=%d: %.2e', t, likelihood);
        likelihood = 1e-300;  % Prevent log(0)
    end
end

%% ========================================================================
%% INTERNAL FUNCTION: UPDATE STEP
%% ========================================================================

function pi_updated = update_step_internal(pi_pred, densities, likelihood)
% UPDATE_STEP  Bayesian update using observed y_t
%
% Reference: Equation (20), page 133

    %% Bayes' rule (equation 20)
    pi_updated = (densities .* pi_pred) / likelihood;
    
    %% Normalize (to handle numerical errors)
    sum_pi = sum(pi_updated);
    if abs(sum_pi - 1) > 1e-10
        pi_updated = pi_updated / sum_pi;
    end
    
    %% Validation
    if any(pi_updated < 0) || any(isnan(pi_updated))
        warning('Invalid probabilities in update step');
        pi_updated = max(0, pi_updated);
        pi_updated = pi_updated / sum(pi_updated);
    end
end

%% ========================================================================
%% HELPER FUNCTION
%% ========================================================================

function idx = find_config_index(configs, target_config)
% Find the row index of target_config in configs matrix
% Returns 0 if not found

    n_configs = size(configs, 1);
    idx = 0;
    
    for i = 1:n_configs
        if all(configs(i, :) == target_config)
            idx = i;
            return;
        end
    end
end