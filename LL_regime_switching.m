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
        %% Reconstruct full parameter vector based on model type
        switch lower(model_type)
            
            case 'mean'
                % Mean switches, sigma constant
                % theta = [alpha; tau; rho; mu_low; mu_high; gamma(1:k); sigma]
                % Need to duplicate sigma
                
                if length(theta) ~= (3 + 2 + k + 1)
                    error('Mean model requires %d parameters, got %d', 3+2+k+1, length(theta));
                end
                
                theta_full = [
                    theta(1:3);              % alpha, tau, rho
                    theta(4:5);              % mu_low, mu_high
                    theta(6:5+k);            % gamma(1:k)
                    theta(end);              % sigma_low (same as sigma)
                    theta(end)               % sigma_high (duplicate)
                ];
                
            case 'vol'
                % Volatility switches, mu constant
                % theta = [alpha; tau; rho; mu; gamma(1:k); sigma_low; sigma_high]
                % Need to duplicate mu
                
                if length(theta) ~= (3 + 1 + k + 2)
                    error('Vol model requires %d parameters, got %d', 3+1+k+2, length(theta));
                end
                
                theta_full = [
                    theta(1:3);              % alpha, tau, rho
                    theta(4);                % mu (same for both regimes)
                    theta(4);                % mu (duplicate)
                    theta(5:4+k);            % gamma(1:k)
                    theta(5+k:end)           % sigma_low, sigma_high
                ];
                
            case 'both'
                % Both mu and sigma switch
                % theta = [alpha; tau; rho; mu_low; mu_high; gamma(1:k); sigma_low; sigma_high]
                % Already in correct format
                
                if length(theta) ~= (3 + 2 + k + 2)
                    error('Both model requires %d parameters, got %d', 3+2+k+2, length(theta));
                end
                
                theta_full = theta;
                
            case 'exogenous'
                % Both switch, but force rho = 0
                % theta = [alpha; tau; mu_low; mu_high; gamma(1:k); sigma_low; sigma_high]
                % Insert rho = 0
                
                if length(theta) ~= (2 + 2 + k + 2)
                    error('Exogenous model requires %d parameters, got %d', 2+2+k+2, length(theta));
                end
                
                theta_full = [
                    theta(1:2);              % alpha, tau
                    0;                       % rho = 0 (forced)
                    theta(3:end)             % mu_low, mu_high, gamma, sigmas
                ];
                
            case 'mean_exo'
                % Mean switches, sigma constant, rho = 0
                % theta = [alpha; tau; mu_low; mu_high; gamma(1:k); sigma]
                
                if length(theta) ~= (2 + 2 + k + 1)
                    error('Mean exogenous model requires %d parameters, got %d', 2+2+k+1, length(theta));
                end
                
                theta_full = [
                    theta(1:2);              % alpha, tau
                    0;                       % rho = 0
                    theta(3:4);              % mu_low, mu_high
                    theta(5:4+k);            % gamma(1:k)
                    theta(end);              % sigma_low
                    theta(end)               % sigma_high (duplicate)
                ];
                
            case 'vol_exo'
                % Volatility switches, mu constant, rho = 0
                % theta = [alpha; tau; mu; gamma(1:k); sigma_low; sigma_high]
                
                if length(theta) ~= (2 + 1 + k + 2)
                    error('Vol exogenous model requires %d parameters, got %d', 2+1+k+2, length(theta));
                end
                
                theta_full = [
                    theta(1:2);              % alpha, tau
                    0;                       % rho = 0
                    theta(3);                % mu
                    theta(3);                % mu (duplicate)
                    theta(4:3+k);            % gamma(1:k)
                    theta(4+k:end)           % sigma_low, sigma_high
                ];
                
            otherwise
                error('Unknown model_type: %s. Use ''mean'', ''vol'', ''both'', ''exogenous'', ''mean_exo'', or ''vol_exo''.', model_type);
        end
        
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