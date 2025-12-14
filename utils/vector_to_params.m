function params = vector_to_params(theta_vec, k, model_type)
%% FUNCTION DESCRIPTION
% VECTOR_TO_PARAMS  Converts optimization vector to structure of parameters
% Refer to crate_params.m for more details
% Matches the ordering used in optimize_model.m

    if nargin < 3, model_type = 'mean'; end
    
    is_volatility = strcmpi(model_type, 'volatility');
    
    %% Unpack based on Model Type 
    if is_volatility
        % Volatility model
        % Vector: [mu, sigma_low, sigma_high, alpha, tau, rho, gamma...]
        mu_val     = theta_vec(1);
        sigma_vals = theta_vec(2:3); 
        alpha      = theta_vec(4);
        tau        = theta_vec(5);
        rho        = theta_vec(6);
        gamma      = theta_vec(7:end);
        
        mu_out    = mu_val; 
        sigma_out = sigma_vals;
        r = 2;
    else
        % Mean Model
        % Vector: [mu_low, mu_high, sigma, alpha, tau, rho, gamma...]
        mu_vals    = theta_vec(1:2); 
        sigma_val  = theta_vec(3);
        alpha      = theta_vec(4);
        tau        = theta_vec(5);
        rho        = theta_vec(6);
        gamma      = theta_vec(7:end);
        
        mu_out    = mu_vals;
        sigma_out = sigma_val; 
        r = 2;
    end

    %% Build structure
    params.alpha = alpha;
    params.tau   = tau;
    params.rho   = rho;
    params.mu    = mu_out;
    params.sigma = sigma_out;
    params.gamma = gamma;
    params.k     = k;
    params.r     = r;
    
    % Add convenience fields for explicit low/high access
    if is_volatility
        params.mu_low = mu_out; params.mu_high = mu_out;
        params.sigma_low = sigma_out(1); params.sigma_high = sigma_out(2);
    else
        params.mu_low = mu_out(1); params.mu_high = mu_out(2);
        params.sigma_low = sigma_out; params.sigma_high = sigma_out;
    end
end