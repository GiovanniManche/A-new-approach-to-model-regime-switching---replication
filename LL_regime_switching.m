function neg_log_L = LL_regime_switching(theta, y, k, r, model_type)
%% OBJECTIVE FUNCTION TO MINIMISE
%
% INPUTS:
%   theta      - Parameter vector (size depends on model_type)
%   y          - Time series data (T x 1)
%   k          - AR order
%   r          - Number of regimes (typically 2)
%   model_type - String specifying model:
%                'mean'     : mu switches, sigma constant (10 params if k=4)
%                'vol'      : mu constant, sigma switches (10 params if k=4)
%                'both'     : mu and sigma switch (11 params if k=4)
%                'exogenous': force rho=0, otherwise like 'both'
%
% OUTPUT:
%   neg_log_L  - Negative log-likelihood (for minimization)
%
% PARAMETER ORDERING:
%   theta = [alpha; tau; rho; mu_low; mu_high; gamma(1:k); sigma_low; sigma_high]
%           or subset depending on model_type

    % Set default model type
    if nargin < 5
        model_type = 'both';
    end
    
    try
        % Retrieve augmented set of parameters
        theta_full = retrieve_params(theta, k, model_type);
        
        %% Convert to params structure and evaluate filter
        params = vector_to_params(theta_full, k, r);
        [log_L, ~, ~] = ms_threshold_filter(y, params);
        
        %% Check validity and return
        if isnan(log_L) || isinf(log_L) || log_L < -1e6
            neg_log_L = 1e10;  % Penalty for invalid likelihood
        else
            neg_log_L = -log_L;
        end
        
    catch ME
        % If any error occurs, return penalty
        warning('Error in LL_regime_switching: %s', ME.message);
        neg_log_L = 1e10;
    end
end