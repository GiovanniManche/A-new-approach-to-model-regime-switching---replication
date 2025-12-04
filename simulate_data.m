function [y, s, w, u] = simulate_data(n, params)
%%  FUNCTION DESCRIPTION
    % Simulate data with a endogenous regime switching model
    % Inputs:
    %   - n: number of observations to generate
    %   - params: structure of parameters 
    % Outputs:
    %   - y: observations (n × 1)
    %   - s: states (n × 1), s_t ∈ {0, 1}
    %   - w: latent factor (n × 1)
    %   - u: observed innovation (n × 1)
%% INITIALISATION
    alpha = params.alpha;
    tau = params.tau;
    rho = params.rho;
    mu    = params.mu(:);      % (r × 1) vector
    sigma = params.sigma(:);   % (r × 1) vector
    gamma = params.gamma(:);   % (k × 1) vector
    k = params.k;
    
    y = zeros(n, 1);
    s = zeros(n, 1);
    w = zeros(n, 1);
    u = zeros(n, 1);
    v = zeros(n, 1);
    
    %% FIX AT t=1
    
    % w_0 value depending on alpha
    if abs(alpha) < 1
        % Stationary case: draw from the stationary distribution
        sigma_stat = 1 / sqrt(1 - alpha^2);
        w(1) = normrnd(0, sigma_stat);
    else
        % Random walk: w_0 = 0 for identification
        w(1) = 0;
    end
    
    % Initial state determination
    s(1) = double(w(1) >= tau);  % s(1) ∈ {0, 1}
    
    % Extract regime-dependent parameters at t=1
    mu_1    = mu(s(1) + 1);      % +1 for MATLAB indexing
    sigma_1 = sigma(s(1) + 1);
    
    % Generate first observation
    u(1) = normrnd(0, 1);
    y(1) = mu_1 + sigma_1 * u(1);
    
    %% MAIN LOOP: GENERATE OTHER OBSERVATIONS
    for t = 2:n
        
        %% Step 1: Generate latent factor innovation v_t (endogenous)
        epsilon = normrnd(0, 1);
        v(t) = rho * u(t-1) + sqrt(1 - rho^2) * epsilon;
        
        %% Step 2: Update latent factor
        w(t) = alpha * w(t-1) + v(t);
        
        %% Step 3: Determine current regime
        s(t) = double(w(t) >= tau);  % s(t) ∈ {0, 1}
        
        %% Step 4: Extract regime-dependent parameters
        mu_t    = mu(s(t) + 1);
        sigma_t = sigma(s(t) + 1);
        
        %% Step 5: Compute conditional mean with AR terms
        % Model: gamma(L)(y_t - mu_t) = sigma_t * u_t
        % => y_t = mu_t + sum_{j=1}^k gamma_j * (y_{t-j} - mu_{t-j}) + sigma_t * u_t
        
        m_t = mu_t;
        
        if k > 0
            for j = 1:min(k, t-1)
                % Mean at time t-j
                mu_tj = mu(s(t-j) + 1);
                
                % AR coefficient gamma_j
                if length(gamma) >= j
                    gamma_j = gamma(j);
                else
                    % If gamma is shorter, use last value (or error)
                    gamma_j = gamma(end);
                end
                
                % Add AR contribution
                m_t = m_t + gamma_j * (y(t-j) - mu_tj);
            end
        end
        
        %% Step 6: Generate observation
        u(t) = normrnd(0, 1);
        y(t) = m_t + sigma_t * u(t);
        
    end
end