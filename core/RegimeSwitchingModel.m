classdef RegimeSwitchingModel
    %% CLASS DESCRIPTION
    % Class that encapsulates the endogenous regime switching logic based 
    % on Chang, Choi and Park (2017). 
    %% ====================================================================

    properties
        alpha   % AR coefficient of the latent factor
        tau     % Threshold
        rho     % Endogeneity
    end

    methods
        function obj = RegimeSwitchingModel(alpha, tau, rho)
            %% FUNCTION DESCRIPTION
            % Function that checks if parameters are valid, and assign them
            % as attributes.
            %% ============================================================
            % Validation of parameters: alpha in ]-1,1], rho in ]-1,1[
            if alpha <= -1 || alpha > 1
                if alpha ~= 1, warning('Alpha is outside (-1, 1].'); end
            end
            if rho <= -1 || rho >= 1
                error('Rho must be in (-1, 1).'); 
            end
            
            obj.alpha = alpha;
            obj.tau   = tau;
            obj.rho   = rho;
        end
        %% ================================================================
        function omega = compute_omega_rho(obj, s_prev, u_prev, t)
            %% FUNCTION DESCRIPTION
            % Calculates transition probabilities (theorem 3.1, corollary
            % 3.2)
            %% ============================================================
            % Setup Parameters & Tolerances 
            alpha_val = obj.alpha;
            tau_val   = obj.tau;
            rho_val   = obj.rho;
            
            tol = 1e-6;
            absrho = abs(rho_val);

            % 2. Integrands
            
            % For Case 6 (Stationary):
            fun_statio = @(x) normcdf((tau_val - rho_val*u_prev - ...
                            alpha_val.*x./sqrt(1-alpha_val^2))...
                            ./sqrt(1-rho_val^2), 0, 1) .* normpdf(x);

            % For Case 5 (Unit Root):
            fun_rw = @(x) normcdf((tau_val - rho_val*u_prev ...
                            - x.*sqrt(t-1))./sqrt(1-rho_val^2), 0, 1)...
                            .* normpdf(x);

            % Helper elements for specific analytic cases
            if abs(alpha_val) < 1 - tol
                num_elem = normcdf((tau_val - rho_val*u_prev)*sqrt(1-alpha_val^2)/alpha_val, 0, 1);
                denom_elem = normcdf(tau_val*sqrt(1-alpha_val^2), 0, 1);
            else
                num_elem = 0; 
                denom_elem = 1; 
            end

            % 3. Case Analysis
            
            % CASE 1: |rho| = 1 and alpha = 0
            if abs(absrho - 1) < tol && abs(alpha_val) < tol
                omega = double(rho_val * u_prev < tau_val);
                
            % CASE 2: |rho| = 1 and 0 < alpha < 1
            elseif abs(absrho - 1) < tol && (alpha_val > tol && alpha_val < 1 - tol)
                omega = (1 - s_prev) * min(1, num_elem/denom_elem) ...
                      + s_prev * max(0, (num_elem - denom_elem)/(1 - denom_elem));
                
            % CASE 3: |rho| = 1 and -1 < alpha < 0
            elseif abs(absrho - 1) < tol && (alpha_val < -tol && alpha_val > -1 + tol)
                omega = s_prev * min(1, (1 - num_elem)/(1 - denom_elem)) ...
                      + (1 - s_prev) * max(0, (denom_elem - num_elem)/denom_elem);
                
            % CASE 4: |rho| = 1 and alpha = 1
            elseif abs(absrho - 1) < tol && abs(alpha_val - 1) < tol
                if t == 1
                    omega = normcdf(tau_val - rho_val * u_prev, 0, 1);
                else
                    if rho_val * u_prev > 0
                        omega = 1 - s_prev;
                    else
                        bound = tau_val / sqrt(t - 1);
                        num = normcdf((tau_val - rho_val*u_prev)/sqrt(t-1), 0, 1) ...
                            - s_prev * normcdf(bound, 0, 1);
                        denom = (1 - s_prev) * normcdf(bound, 0, 1) ...
                              + s_prev * (1 - normcdf(bound, 0, 1));
                        omega = num / denom;
                    end
                end
                
            % CASE 5: alpha = 1 and |rho| < 1
            elseif abs(alpha_val - 1) < tol && absrho < 1 - tol
                if t == 1
                    omega = normcdf(tau_val, 0, 1);
                else
                    bound = tau_val / sqrt(t - 1);
                    denom = (1 - s_prev) * normcdf(bound, 0, 1) ...
                          + s_prev * (1 - normcdf(bound, 0, 1));
                    
                    int_low = integral(fun_rw, -Inf, bound);
                    int_high = integral(fun_rw, bound, Inf);
                    
                    omega = ((1 - s_prev) * int_low + s_prev * int_high) / denom;
                end
                
            % CASE 6: |alpha| < 1 and |rho| < 1 (Standard Stationary Case)
            elseif abs(alpha_val) < 1 - tol && absrho < 1 - tol
                bound = tau_val * sqrt(1 - alpha_val^2);
                denom = (1 - s_prev) * normcdf(bound, 0, 1) ...
                      + s_prev * (1 - normcdf(bound, 0, 1));
                
                int_low = integral(fun_statio, -Inf, bound);
                int_high = integral(fun_statio, bound, Inf);
                
                omega = ((1 - s_prev) * int_low + s_prev * int_high) / denom;
                
            % ERROR
            else
                % Fallback or error
                 error('RegimeSwitchingModel: Invalid parameter combination: alpha=%.4f, rho=%.4f', alpha_val, rho_val);
            end
            
            % Sanity check [0, 1]
            if omega < 0, omega = 0; elseif omega > 1, omega = 1; end
        end
    end

end
