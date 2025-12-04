function omega = transition_proba(s_prev, u_prev, t, params)
% COMPUTE_OMEGA_RHO  Transition probability to regime 0 (low state)
%
% Reference: Chang, Choi & Park (2017), Theorem 3.1, Corollary 3.2
%
% Inputs:
%   s_prev : previous state (0 or 1)
%   u_prev : standardized residual at t-1
%   t      : current time
%   params : parameter structure
%
% Output:
%   omega : P(s_t = 0 | s_{t-1}, u_{t-1}, theta)

    %% Validation
    if params.r ~= 2
        error('compute_omega_rho only supports r=2 regimes, got r=%d', params.r);
    end
    
    if s_prev ~= 0 && s_prev ~= 1
        error('s_prev must be 0 or 1, got %d', s_prev);
    end
    
    %% Extract parameters
    alpha = params.alpha;
    tau   = params.tau;
    rho   = params.rho;
    
    %% Numerical tolerance for boundary cases
    tol = 1e-6;
    
    %% Compute transition probability
    absrho = abs(rho);
    
    % Precompute elements (if needed for special cases)
    if abs(alpha) < 1 - tol
        num_elem = normcdf((tau - rho*u_prev)*sqrt(1-alpha^2)/alpha, 0, 1);
        denom_elem = normcdf(tau*sqrt(1-alpha^2), 0, 1);
    end
    
    % Define integrands with ELEMENTWISE operations (.* ./ .^)
    fun_statio = @(x) normcdf((tau - rho*u_prev - alpha.*x./sqrt(1-alpha^2))./sqrt(1-rho^2), 0, 1) .* normpdf(x);
    
    fun_rw = @(x) normcdf((tau - rho*u_prev - x.*sqrt(t-1))./sqrt(1-rho^2), 0, 1) .* normpdf(x);
    
    %% CASE ANALYSIS
    
    % CASE 1: |rho| = 1 and alpha = 0
    if abs(absrho - 1) < tol && abs(alpha) < tol
        omega = double(rho * u_prev < tau);
        
    % CASE 2: |rho| = 1 and 0 < alpha < 1
    elseif abs(absrho - 1) < tol && (alpha > tol && alpha < 1 - tol)
        omega = (1 - s_prev) * min(1, num_elem/denom_elem) ...
              + s_prev * max(0, (num_elem - denom_elem)/(1 - denom_elem));
        
    % CASE 3: |rho| = 1 and -1 < alpha < 0
    elseif abs(absrho - 1) < tol && (alpha < -tol && alpha > -1 + tol)
        omega = s_prev * min(1, (1 - num_elem)/(1 - denom_elem)) ...
              + (1 - s_prev) * max(0, (denom_elem - num_elem)/denom_elem);
        
    % CASE 4: |rho| = 1 and alpha = 1
    elseif abs(absrho - 1) < tol && abs(alpha - 1) < tol
        if t == 1
            omega = normcdf(tau - rho * u_prev, 0, 1);
        else
            if rho * u_prev > 0
                omega = 1 - s_prev;
            else
                bound = tau / sqrt(t - 1);
                num = normcdf((tau - rho*u_prev)/sqrt(t-1), 0, 1) ...
                    - s_prev * normcdf(bound, 0, 1);
                denom = (1 - s_prev) * normcdf(bound, 0, 1) ...
                      + s_prev * (1 - normcdf(bound, 0, 1));
                omega = num / denom;
            end
        end
        
    % CASE 5: alpha = 1 and |rho| < 1
    elseif abs(alpha - 1) < tol && absrho < 1 - tol
        if t == 1
            omega = normcdf(tau, 0, 1);
        else
            bound = tau / sqrt(t - 1);
            denom = (1 - s_prev) * normcdf(bound, 0, 1) ...
                  + s_prev * (1 - normcdf(bound, 0, 1));
            
            int_low = integral(fun_rw, -Inf, bound);
            int_high = integral(fun_rw, bound, Inf);
            
            omega = ((1 - s_prev) * int_low + s_prev * int_high) / denom;
        end
        
    % CASE 6: |alpha| < 1 and |rho| < 1 (MOST COMMON)
    elseif abs(alpha) < 1 - tol && absrho < 1 - tol
        bound = tau * sqrt(1 - alpha^2);
        denom = (1 - s_prev) * normcdf(bound, 0, 1) ...
              + s_prev * (1 - normcdf(bound, 0, 1));
        
        int_low = integral(fun_statio, -Inf, bound);
        int_high = integral(fun_statio, bound, Inf);
        
        omega = ((1 - s_prev) * int_low + s_prev * int_high) / denom;
        
    % ERROR: Invalid parameters
    else
        error('Invalid parameter combination: alpha=%.10f, rho=%.10f', alpha, rho);
    end
    
    % Ensure omega is in [0, 1]
    omega = max(0, min(1, omega));
end