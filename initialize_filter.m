function pi_1 = initialize_filter(params)
% INITIALIZE_FILTER  Compute initial distribution over state configurations
%
% Computes pi_1 = P(s_1, s_0, ..., s_{1-k} | F)
% i.e., the initial probability distribution at t=1
%
% Inputs:
%   params : parameter structure with fields:
%            - alpha : AR coefficient of latent factor
%            - tau   : threshold
%            - rho   : endogeneity parameter
%            - k     : AR order
%            - r     : number of regimes (must be 2)
%
% Output:
%   pi_1 : (r^(k+1) × 1) vector of initial probabilities
%          pi_1(i) = P(config_i at t=1)
%
% Note: This assumes we observe y_1, so we need the joint distribution
%       of (s_1, s_0, ..., s_{1-k})

    %% Validation
    if params.r ~= 2
        error('initialize_filter only supports r=2 regimes');
    end
    
    %% Extract parameters
    alpha = params.alpha;
    tau = params.tau;
    k = params.k;
    r = params.r;
    
    tol = 1e-6;  % Numerical tolerance
    
    %% Generate all configurations
    configs = generate_all_configs(k, r);
    n_configs = size(configs, 1);  % r^(k+1)
    
    %% Initialize output
    pi_1 = zeros(n_configs, 1);
    
    %% STEP 1: Compute marginal distribution of s_0
    % P(s_0 = 0) and P(s_0 = 1)
    
    if abs(alpha - 1) < tol
        % Case: Random walk (alpha = 1)
        % We set w_0 = 0 for identification
        if tau > 0
            prob_s0_low = 1;      % P(s_0 = 0) = P(w_0 < tau) = P(0 < tau) = 1
            prob_s0_high = 0;     % P(s_0 = 1) = 0
        else
            prob_s0_low = 0;      % P(s_0 = 0) = P(0 < tau) = 0
            prob_s0_high = 1;     % P(s_0 = 1) = 1
        end
        
    else
        % Case: Stationary (|alpha| < 1)
        % w_0 ~ N(0, 1/(1-alpha^2))
        bound = tau * sqrt(1 - alpha^2);
        prob_s0_low = normcdf(bound, 0, 1);      % P(s_0 = 0)
        prob_s0_high = 1 - prob_s0_low;          % P(s_0 = 1)
    end
    
    %% STEP 2: For each configuration, compute joint probability
    % P(s_1, s_0, ..., s_{1-k}) = P(s_1 | s_0, ..., s_{1-k}) × P(s_0, ..., s_{1-k})
    %
    % For simplicity at t=1, we assume:
    % - If k=0: Just compute P(s_1)
    % - If k>=1: Compute P(s_1, s_0) using transition from s_0 to s_1
    %           For s_{-1}, s_{-2}, ... use marginal distribution recursively
    
    for i = 1:n_configs
        config = configs(i, :);  % [s_1, s_0, s_{-1}, ..., s_{1-k}]
        
        if k == 0
            % No lags, just s_1
            s_1 = config(1);
            if s_1 == 0
                pi_1(i) = prob_s0_low;   % Reuse as P(s_1=0)
            else
                pi_1(i) = prob_s0_high;
            end
            
        else
            % k ≥ 1: Need to compute joint distribution
            
            % Start with P(s_0)
            s_0 = config(2);
            if s_0 == 0
                prob = prob_s0_low;
            else
                prob = prob_s0_high;
            end
            
            % For s_{-1}, ..., s_{1-k}, assume marginal distribution
            % (since we don't have data before t=0)
            for j = 3:length(config)
                s_past = config(j);
                if s_past == 0
                    prob = prob * prob_s0_low;
                else
                    prob = prob * prob_s0_high;
                end
            end
            
            % Now multiply by P(s_1 | s_0, ..., s_{1-k})
            % For t=1, we don't have u_0, so we use exogenous transition
            % (assumes rho=0 effectively at initialization)
            
            s_1 = config(1);
            omega = compute_transition_exogenous(s_0, params);
            
            if s_1 == 0
                prob = prob * omega;        % Transition to low
            else
                prob = prob * (1 - omega);  % Transition to high
            end
            
            pi_1(i) = prob;
        end
    end
    
    %% STEP 3: Normalize (to ensure sum = 1, accounting for numerical errors)
    pi_1 = pi_1 / sum(pi_1);
    
    %% Validation
    if abs(sum(pi_1) - 1) > 1e-10
        warning('Initial distribution does not sum to 1: sum = %.10f', sum(pi_1));
    end
    
    if any(pi_1 < 0)
        error('Negative probabilities detected in initial distribution');
    end
end

%% ========================================================================
%% HELPER FUNCTION: Exogenous transition (rho = 0)
%% ========================================================================

function omega = compute_transition_exogenous(s_prev, params)
% Compute transition probability with rho = 0 (exogenous)
% This uses the closed-form formulas from Lemma 2.1
    
    alpha = params.alpha;
    tau = params.tau;
    tol = 1e-6;
    
    if abs(alpha - 1) < tol
        % Random walk case
        if tau > 0
            if s_prev == 0
                omega = 1;  % P(s_1=0 | s_0=0) = 1 if tau > 0
            else
                omega = 0;  % P(s_1=0 | s_0=1) = 0 if tau > 0
            end
        else
            if s_prev == 0
                omega = 0;
            else
                omega = 1;
            end
        end
        
    else
        % Stationary case (use Lemma 2.1 formulas)
        bound = tau * sqrt(1 - alpha^2);
        
        if s_prev == 0
            % P(s_1 = 0 | s_0 = 0)
            % a(alpha, tau) from Lemma 2.1
            integrand = @(x) normcdf((tau - alpha*x/sqrt(1-alpha^2)), 0, 1) ...
                             .* normpdf(x);
            omega = integral(integrand, -Inf, bound) / normcdf(bound, 0, 1);
            
        else
            % P(s_1 = 0 | s_0 = 1)
            % 1 - b(alpha, tau) from Lemma 2.1
            integrand = @(x) normcdf((tau - alpha*x/sqrt(1-alpha^2)), 0, 1) ...
                             .* normpdf(x);
            omega = integral(integrand, bound, Inf) / (1 - normcdf(bound, 0, 1));
        end
    end
    
    omega = max(0, min(1, omega));  % Ensure [0,1]
end