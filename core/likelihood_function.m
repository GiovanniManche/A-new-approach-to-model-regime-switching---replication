function [nloglik, filtered_probs_final] = likelihood_function(params, y, k, is_vol_model)
    %% FUNCTION DESCRIPTION
    % Computes the likelihood function, using for that the Chang, Choi and 
    % Park (2017) modified Markov filter with an predicting and updating 
    % steps. 
    % 
    % INPUTS:
    %   params       : VECTOR (!!) containing the parameters of the model
    %   y            : time series on which we estimate the model
    %   k            : number of AR lags for y
    %   is_vol_model : boolean to account for the fact that we either
    %   estimate a mean or a volatility model (to know at which place
    %   parameters are).
    % 
    % OUTPUTS:
    %   nloglik: value of the log-likelihood
    %   filtered_probs_final: filtered probabilities
    %% ====================================================================
    %% 1. Unpack Parameters 
    % Common indices: Alpha(4), Tau(5), Rho(6), Gamma(7:end)
    
    alpha = params(4); 
    tau   = params(5); 
    rho   = params(6); 
    gamma = params(7:end);

    if is_vol_model
        % Volatility Model: [mu, sig_L, sig_H, ...]
        mu_common = params(1);
        mu_low = mu_common; 
        mu_high = mu_common;
        
        sigma_low  = params(2); 
        sigma_high = params(3);
    else
        % Mean Model: [mu_L, mu_H, sig, ...]
        mu_low  = params(1); 
        mu_high = params(2);
        
        sigma_common = params(3);
        sigma_low  = sigma_common; 
        sigma_high = sigma_common;
    end

    % Constraints check 
    if sigma_low <= 1e-4 || sigma_high <= 1e-4 || abs(rho) >= 0.999 || abs(alpha) > 1
        nloglik = 1e10; filtered_probs_final = []; return;
    end

    %% 2. Setup Filter 
    % We generate the different configurations of s_{t},..., s_{t-k} in
    % order to iterate on them afterwards
    configs = generate_configs(k); 
    N_configs = size(configs, 1);
    
    mus = [mu_low, mu_high];
    sigs = [sigma_low, sigma_high];
    
    % Creates a object for the model
    model = RegimeSwitchingModel(alpha, tau, rho);
    % At start, attributes a uniform probability to be in each
    % configuration s_t,..., s_{t-k}
    prob_state = ones(N_configs, 1) / N_configs; 
    

    T = length(y);
    eff_T = T - k;
    loglik = 0;
    filtered_probs_final = zeros(1, eff_T);

    gamma = gamma(:); 
    
    %% 3. Filter Loop
    for t = 1:eff_T
        idx = k + t;
        densities = zeros(N_configs, 1);
        % Will contain the probabilities of each configuration
        prob_predict = zeros(N_configs, 1);
        
        %% PREDICTION STEP: p(s_t,..., s_{t-k}|F_{t-1})
        for j = 1:N_configs
            % Pruning: if the probability of one configuration is too low,
            % ignore it. Allows the optimisation to be faster and more
            % robust.
            if prob_state(j) < 1e-12, continue; end
            
            % Get the previous configuration and state.
            cfg_prev = configs(j, :);
            s_prev = cfg_prev(1);
            
            % Get mu and sigma of the previous period (to standardise
            % u_{t-1}
            idx_s_prev = round(s_prev) + 1;
            mu_prev = mus(idx_s_prev);
            sig_prev = sigs(idx_s_prev);
            
            % Now, we compute u_{t-1} the innovation of the AR model
            % Because it is correlated with v_t, the innovation on the
            % latent AR model.
            % At start, u_0 = 0
            if t == 1
                u_prev_val = 0; 
            else
                ar_sum_prev = 0;
                for lag = 1:k
                    s_lag = cfg_prev(lag + 1); % [S_{t-1},..., S_{t-k}]
                    y_idx = idx - 1 - lag;     % get y_{t-1-k}
                    % Compute gamma_k x (y_{t-1-k} - mu_{s_{t-1-k}}
                    ar_sum_prev = ar_sum_prev + gamma(lag) * (y(y_idx) - mus(round(s_lag)+1));
                end
                % Compute standardised residuals
                resid_prev = (y(idx-1) - mu_prev) - ar_sum_prev;
                u_prev_val = resid_prev / sig_prev;
            end
            
            % Compute omega_rho (for p(s_t|s_t-1,...))
            % Dependencies to previous k states via u_{t-1}
            omega = model.compute_omega_rho(s_prev, u_prev_val, t);
            
            % Next configurations,  go from t-1 to t
            % We don't know S_t yet so two possibilites: S_t = {0,1}
            cfg_next_0 = [0, cfg_prev(1:k)];

            % Transform a binary vector [1,0,1..] into a line number, cf
            % below
            idx_0 = binary_search_config(cfg_next_0);
            
            cfg_next_1 = [1, cfg_prev(1:k)];
            idx_1 = binary_search_config(cfg_next_1);
            
            % The "current" probability mass has to be distributed between 
            % two future possible configurations (s_t = 0 or s_t = 1) with
            % probability omega for s_t = 0 and (1-omega) for s_t = 1
            prob_predict(idx_0) = prob_predict(idx_0) + omega * prob_state(j);
            prob_predict(idx_1) = prob_predict(idx_1) + (1-omega) * prob_state(j);
        end
        
        %% UPDATE STEP: p(s_t,...,s_{t-k}|F_t)
        % Simple Bayes rule. The loops' logic is the same as in the
        % prediction step. 
        for i = 1:N_configs
            if prob_predict(i) < 1e-12, continue; end
            
            cfg_curr = configs(i, :);
            s_curr = cfg_curr(1);
            
            idx_s_curr = round(s_curr) + 1;
            mu_curr = mus(idx_s_curr);
            sig_curr = sigs(idx_s_curr);
            
            ar_sum = 0;
            for lag = 1:k
                s_lag = cfg_curr(lag + 1);
                ar_sum = ar_sum + gamma(lag) * (y(idx - lag) - mus(round(s_lag)+1));
            end
            % Compute p(y_t|s_t,..., s_{t-k}, F_{t-1}) = N(m_t, sigma_t)
            res_t = (y(idx) - mu_curr) - ar_sum;
            densities(i) = (1/sig_curr) * exp(-0.5 * (res_t/sig_curr)^2) / sqrt(2*pi);
        end
        
        % Compute the likelihood 
        f_y = sum(densities .* prob_predict);
        if f_y < 1e-20, f_y = 1e-20; end
        loglik = loglik + log(f_y);

        % Compute p(s_t,..., s_{t-k}|F_t)
        prob_state = (densities .* prob_predict) / f_y;

        % The filter tracks joint probabilities of entire paths (St, St-1, ..., St-k).
        % To get the simple probability P(St = 1 | Data_t), we must marginalize
        % the probabilities of ALL configurations where the current state (col 1) is 1.
        % We compute here P(S_t = 1|F_t)
        mask_high = (configs(:,1) == 1);
        filtered_probs_final(t) = sum(prob_state(mask_high));
    end
    
    nloglik = -loglik;
    if isnan(nloglik) || isinf(nloglik), nloglik = 1e10; end
end

function idx = binary_search_config(target)
    % Helper function that transforms a binary vector in an integer to directly access
    % probability matrices. 
    str = char(target + '0');
    idx = bin2dec(str) + 1;
end