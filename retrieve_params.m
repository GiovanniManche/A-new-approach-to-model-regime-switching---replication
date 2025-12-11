function theta_full = retrieve_params(theta,k,model_type)
%Function to compute full parameters vectors depending on model type
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
end

