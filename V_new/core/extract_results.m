function [results] = extract_results(params, y, k, model_type)
    %% FUNCTION DESCRIPTION
    % Re-runs the filter on optimal parameters to extract detailed time 
    % series. This function runs the Regime Switching filter one last time 
    % using the estimated parameters (params) to recover the "hidden" 
    % trajectories of the model, such as the probability of being in a high
    % /low regime, and the time-varying transition probabilities 
    % driven by endogeneity.
    %
    % INPUTS:
    %   params     : Vector of estimated parameters (output from optimization).
    %                Structure depends on model_type (Mean or Volatility).
    %   y          : Vector of observed data (T x 1).
    %   k          : Number of autoregressive lags
    %   model_type : String 'mean' or 'volatility'.
    %
    % OUTPUTS:
    %   results.filtered_probs : (T x 1) Vector of probabilities P(S_t = 1 | I_t).
    %                            Probability of being in the "High" regime (Expansion or High Vol).
    %   results.p00            : (T x 1) Time-varying probability of staying in Regime 0.
    %                            P(S_t=0 | S_{t-1}=0, u_{t-1}).
    %   results.p11            : (T x 1) Time-varying probability of staying in Regime 1.
    %                            P(S_t=1 | S_{t-1}=1, u_{t-1}).
    %   results.latent_w       : (T x 1) Estimated trajectory of the latent variable E[w_t].
    %                            Useful to visualize the latent process.
    %
    % NOTES:
    %   - Unlike likelihood_function.m, this function is not optimized for speed but for
    %     data extraction. It stores intermediate values like sum_p00 or w_expected.    
    %   - The 'w_expected' is an approximation since the true path of w_t is unobservable.
    %% ====================================================================

    % Extract parameters
    is_volatility = strcmpi(model_type, 'volatility');    
    alpha = params(4);
    tau   = params(5);
    rho   = params(6);
    gamma = params(7:end);
    
    if is_volatility
        mu_common = params(1);
        mu_low = mu_common; mu_high = mu_common;
        sigma_low = params(2); sigma_high = params(3);
    else
        mu_low = params(1); mu_high = params(2);
        sigma_common = params(3);
        sigma_low = sigma_common; sigma_high = sigma_common;
    end
    
    mus = [mu_low, mu_high];
    sigs = [sigma_low, sigma_high];
    
    % Creates possible configurations.
    configs = generate_configs(k);
    N_configs = size(configs, 1);
    
    % Creates a model object and assign a uniform probability to each
    % potential configuration at start.
    model = RegimeSwitchingModel(alpha, tau, rho);
    prob_state = ones(N_configs, 1) / N_configs;
    
    T = length(y);
    eff_T = T - k;
    
    filtered_probs = zeros(eff_T, 1);
    % p00 = P(S_t= 0 | S_t-1 = 0)
    p00_series = zeros(eff_T, 1);
    % p11 = P(S_t = 1 | S_t-1 = 1)
    p11_series = zeros(eff_T, 1);
    
    % Latent factor approximation
    % w_t = alpha * w_{t-1} + rho * u_{t-1} + e_t
    % We will track E[w_t]
    w_expected = zeros(eff_T + 1, 1); 
    % Initialize w_0 (steady state mean is 0)
    w_expected(1) = 0;
    
    gamma = gamma(:);
    
    for t = 1:eff_T
        idx = k + t;
        densities = zeros(N_configs, 1);
        prob_predict = zeros(N_configs, 1);
        
        % Accumulators for transition probs
        sum_p00 = 0; w_p00 = 0;
        sum_p11 = 0; w_p11 = 0;
        
        % Average residual for latent factor update
        avg_u_prev = 0;
        
        % Here, same logic as in likelihood_function, please refer to this
        % module. 
        for j = 1:N_configs
            if prob_state(j) < 1e-12, continue; end
            
            cfg_prev = configs(j, :);
            s_prev = cfg_prev(1);
            
            idx_s_prev = round(s_prev) + 1;
            mu_prev = mus(idx_s_prev);
            sig_prev = sigs(idx_s_prev);
            
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
            
            % Accumulate weighted u_prev
            avg_u_prev = avg_u_prev + prob_state(j) * u_prev_val;
            
            % Compute transition prob P(s_t=0 | ...)
            omega = model.compute_omega_rho(s_prev, u_prev_val, t);
            
            % Store for averaging
            if s_prev == 0
                sum_p00 = sum_p00 + omega * prob_state(j);
                w_p00 = w_p00 + prob_state(j);
            else
                % if s_prev = 1, prob of switching to 0 is omega
                % so prob of staying in 1 is (1 - omega)
                sum_p11 = sum_p11 + (1 - omega) * prob_state(j);
                w_p11 = w_p11 + prob_state(j);
            end
            
            % Propagate
            cfg_next_0 = [0, cfg_prev(1:k)];
            idx_0 = binary_search_config(cfg_next_0);
            
            cfg_next_1 = [1, cfg_prev(1:k)];
            idx_1 = binary_search_config(cfg_next_1);
            
            prob_predict(idx_0) = prob_predict(idx_0) + omega * prob_state(j);
            prob_predict(idx_1) = prob_predict(idx_1) + (1-omega) * prob_state(j);
        end
        
        % Compute averages across configurations for this time step
        if w_p00 > 0, p00_series(t) = sum_p00 / w_p00; else, p00_series(t) = NaN; end
        if w_p11 > 0, p11_series(t) = sum_p11 / w_p11; else, p11_series(t) = NaN; end
        
        % Update latent factor expectation
        % w_t = alpha w_{t-1} + v_t 
        % v_t = rho u_{t-1} - e_t * sqrt{1-rho^2}
        % E[w_t] = alpha * E[w_{t-1}] + rho * E[u_{t-1}]
        w_expected(t+1) = alpha * w_expected(t) + rho * avg_u_prev;
        
        % Update Step
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
            
            res_t = (y(idx) - mu_curr) - ar_sum;
            densities(i) = (1/sig_curr) * exp(-0.5 * (res_t/sig_curr)^2) / sqrt(2*pi);
        end
        
        f_y = sum(densities .* prob_predict);
        if f_y < 1e-20, f_y = 1e-20; end
        
        prob_state = (densities .* prob_predict) / f_y;
        
        mask_high = (configs(:,1) == 1);
        filtered_probs(t) = sum(prob_state(mask_high));
    end
    
    results.filtered_probs = filtered_probs;
    results.p00 = p00_series;
    results.p11 = p11_series;
    results.latent_w = w_expected(2:end); % Align with t=1..T
end

function idx = binary_search_config(target)
    str = char(target + '0');
    idx = bin2dec(str) + 1;
end
