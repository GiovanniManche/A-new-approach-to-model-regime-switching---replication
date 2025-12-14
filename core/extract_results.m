function [results] = extract_results(params, y, k, model_type)
    %% FUNCTION DESCRIPTION 
    % Compute the infered elements of the model: 
    %   - filtered probabilities
    %   - the time-varying probabilities of staying in the low/high state
    %   - the infered latent factor
    % using parameters given (usually those obtained via MLE).
    %
    % INPUTS:
    %   params     : Vector of estimated parameters (output from optimization).
    %                Structure depends on model_type (Mean or Volatility).
    %   y          : Vector of observed data (T x 1).
    %   k          : Number of autoregressive lags
    %   model_type : String 'mean' or 'volatility'.
    %
    % OUTPUTS:
    %   results.filtered_probs : (T x 1) Vector of probabilities P(S_t = 0 | I_t).
    %                            Probability of being in the "Low" regime (Recession or Low Vol).
    %   results.p00            : (T x 1) Time-varying probability of staying in Regime 0.
    %   results.p11            : (T x 1) Time-varying probability of staying in Regime 1.
    %   results.latent_w       : (T x 1) Estimated trajectory of the latent variable (corollary 3.3).
    %% ====================================================================
    
    %% 1. Extract parameters
    % Convert into a boolean
    is_volatility = strcmpi(model_type, 'volatility');    
    alpha = params(4);
    tau   = params(5);
    rho   = params(6);
    gamma = params(7:end);
    
    if is_volatility
        mu_common = params(1);
        mus = [mu_common, mu_common];
        sigma_low = params(2); sigma_high = params(3);
        sigs = [sigma_low, sigma_high];
    else
        mu_low = params(1); mu_high = params(2);
        sigma_common = params(3);
        mus = [mu_low, mu_high];
        sigs = [sigma_common, sigma_common];
    end
    
    % Generate possible configurations (s_t, s_{t-1},..., s_{t-k})
    configs = generate_configs(k);
    N_configs = size(configs, 1);
    
    % Create a Regime Switching structure (cf. documentation for this
    % module)
    model = RegimeSwitchingModel(alpha, tau, rho);
    % Initially, we affect uniform probabilities to all scenarios (diffuse
    % prior)
    prob_state = ones(N_configs, 1) / N_configs;
    
    T = length(y);
    eff_T = T - k;
    gamma = gamma(:);
    
    %% 2. Setup for the latent factor
    % We discretize the continuous latent factor w_t on a finite grid to perform
    % numerical integration in the filtering algorithm.
    % All depends on the persistency of w_t
    if abs(alpha) < 1
        w_std = sqrt(1 / (1 - alpha^2));
        w_min = -4 * w_std;
        w_max = 4 * w_std;
    else
        w_min = -10 * sqrt(eff_T);
        w_max = 10 * sqrt(eff_T);
    end
    
    N_grid = 200;
    w_grid = linspace(w_min, w_max, N_grid)';
    dw = w_grid(2) - w_grid(1);
    
    %% 3. Storage
    filtered_probs = zeros(eff_T, 1);
    p00_series = zeros(eff_T, 1);
    p11_series = zeros(eff_T, 1);
    w_filtered = zeros(eff_T, 1);
    w_variance = zeros(eff_T, 1);
    w_dist_filtered = zeros(eff_T, N_grid);
    
    %% 4. Initialize w_0 
    % If the latent factor is stationary, w_0 ~ N(0, 1/(1-alpha^2)).
    % Otherwise w_0 = 0
    if abs(alpha) < 1
        w_density = normpdf(w_grid, 0, sqrt(1/(1-alpha^2)));
    else
        w_density = zeros(N_grid, 1);
        [~, idx_zero] = min(abs(w_grid));
        w_density(idx_zero) = 1/dw;
    end
    w_density = w_density / (sum(w_density) * dw);
    
    %% 5. LOOPS
    % We loop on time, then, at each time t, run the prediction and update 
    % steps following the CCP filter
    for t = 1:eff_T
        idx = k + t;
        
        % Storage 
        densities = zeros(N_configs, 1);
        prob_predict = zeros(N_configs, 1);
        w_pred_density = zeros(N_configs, N_grid);
        % Accumulators for transition proba
        sum_p00 = 0; w_p00 = 0;
        sum_p11 = 0; w_p11 = 0;
        
        % Here, same logic as in likelihood_function, please refer to this
        % module. 
        for j = 1:N_configs
            if prob_state(j) < 1e-15, continue; end
            
            cfg_prev = configs(j, :);
            s_prev = cfg_prev(1);
            idx_s_prev = round(s_prev) + 1;
            mu_prev = mus(idx_s_prev);
            sig_prev = sigs(idx_s_prev);
            
            % Calculate u_{t-1}
            if t == 1
                u_prev_val = 0;
            else
                ar_sum_prev = 0;
                for lag = 1:k
                    s_lag = cfg_prev(lag + 1);
                    y_idx = idx - 1 - lag;
                    ar_sum_prev = ar_sum_prev + gamma(lag) * (y(y_idx) - mus(round(s_lag)+1));
                end
                resid_prev = (y(idx-1) - mu_prev) - ar_sum_prev;
                u_prev_val = resid_prev / sig_prev;
            end
            
            % Transition probability
            omega = model.compute_omega_rho(s_prev, u_prev_val, t);
            
            if s_prev == 0
                sum_p00 = sum_p00 + omega * prob_state(j);
                w_p00 = w_p00 + prob_state(j);
            else
                sum_p11 = sum_p11 + (1 - omega) * prob_state(j);
                w_p11 = w_p11 + prob_state(j);
            end
            
            % Future configs
            idx_0 = binary_search_config([0, cfg_prev(1:k)]);
            idx_1 = binary_search_config([1, cfg_prev(1:k)]);
            
            prob_predict(idx_0) = prob_predict(idx_0) + omega * prob_state(j);
            prob_predict(idx_1) = prob_predict(idx_1) + (1-omega) * prob_state(j);
            
            %% PREDICTION for w_t (corollary 3.3)
            for w_idx = 1:N_grid
                w_val = w_grid(w_idx);
                % Compute p(w_t|s_{t-1},...)
                w_pred_density(j, w_idx) = transition_density_latent(...
                    w_val, u_prev_val, s_prev, tau, alpha, rho, t);
            end
            
            % Normalize
            w_pred_density(j, :) = w_pred_density(j, :) / (sum(w_pred_density(j, :)) * dw + 1e-20);
        end
        
        % Store transition probabilities
        if w_p00 > 0, p00_series(t) = sum_p00 / w_p00; else, p00_series(t) = NaN; end
        if w_p11 > 0, p11_series(t) = sum_p11 / w_p11; else, p11_series(t) = NaN; end
        
        %% Marginalize over states: p(w_t | F_{t-1})
        w_predict = zeros(N_grid, 1);
        for j = 1:N_configs
            if prob_state(j) > 1e-15
                w_predict = w_predict + w_pred_density(j, :)' * prob_state(j);
            end
        end
        w_predict = w_predict / (sum(w_predict) * dw + 1e-20);
        
        %% UPDATE STEP for states 
        for i = 1:N_configs
            if prob_predict(i) < 1e-15, continue; end
            cfg_curr = configs(i, :);
            idx_s_curr = round(cfg_curr(1)) + 1;
            
            ar_sum = 0;
            for lag = 1:k
                s_lag = cfg_curr(lag + 1);
                ar_sum = ar_sum + gamma(lag) * (y(idx - lag) - mus(round(s_lag)+1));
            end
            
            res_t = (y(idx) - mus(idx_s_curr)) - ar_sum;
            densities(i) = normpdf(res_t, 0, sigs(idx_s_curr));
        end
        
        f_y = sum(densities .* prob_predict);
        if f_y < 1e-20, f_y = 1e-20; end
       
        prob_state = (densities .* prob_predict) / f_y;
        
        mask_high = (configs(:,1) == 1);
       
        filtered_probs(t) = sum(prob_state(mask_high));
        
        %% UPDATE STEP for w_t: compute p(w_t|s_{t-1},..., s_{t-k}, F_t)
        likelihood_w = zeros(N_grid, 1);
        for w_idx = 1:N_grid
            w_val = w_grid(w_idx);
            s_implied = double(w_val >= tau);
            mask_compatible = (configs(:, 1) == s_implied);
            
            lik_sum = 0;
            for i = 1:N_configs
                if mask_compatible(i) && prob_predict(i) > 1e-15
                    cfg_curr = configs(i, :);
                    idx_s_curr = round(cfg_curr(1)) + 1;
                    
                    ar_sum = 0;
                    for lag = 1:k
                        s_lag = cfg_curr(lag + 1);
                        ar_sum = ar_sum + gamma(lag) * (y(idx - lag) - mus(round(s_lag)+1));
                    end
                    
                    res_t = (y(idx) - mus(idx_s_curr)) - ar_sum;
                    lik = normpdf(res_t, 0, sigs(idx_s_curr));
                    lik_sum = lik_sum + lik * prob_predict(i);
                end
            end
            
            likelihood_w(w_idx) = lik_sum;
        end
        
         % Bayes' rule gives p(w_t|s_{t-1},..., s_{t-k}, F_t)
        w_posterior = w_predict .* likelihood_w;
        w_posterior = w_posterior / (sum(w_posterior) * dw + 1e-20);
        
        % Store
        w_dist_filtered(t, :) = w_posterior';
        % p(w_t|F_t) = sum over all possible configurations
        w_filtered(t) = sum(w_grid .* w_posterior) * dw;
        w_variance(t) = sum((w_grid - w_filtered(t)).^2 .* w_posterior) * dw;
        
        w_density = w_posterior;
    end
    
    %% 6. Package results
    results.filtered_probs = filtered_probs;
    results.p00 = p00_series;
    results.p11 = p11_series;
    results.latent_w = w_filtered;
    results.latent_w_variance = w_variance;
    results.w_grid = w_grid;
end

function idx = binary_search_config(target)
    % Helper function that transforms a binary vector in an integer to directly access
    % probability matrices. 
    str = char(target + '0');
    idx = bin2dec(str) + 1;
end