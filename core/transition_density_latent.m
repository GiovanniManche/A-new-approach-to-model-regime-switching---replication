function [pwcond] = transition_density_latent(w, uprec, sprec, tau, alpha, rho, t)
    %% FUNCTION DESCRIPTION
    % Function to compute the transition density of the latent process
    % CORRECTED VERSION - Fixed bugs from original
    % 
    % Inputs :
    % - w: current value for the latent variable
    % - uprec: precedent value for the residual of endogenous process
    % - sprec: precedent state (1 for high state, 0 for low state)
    % - tau: threshold
    % - alpha: autoregressive coefficient for the latent process
    % - rho: correlation coefficient
    % - t: time
    %
    % Output: p(wt|s_{t-1}, F_{t-1})
    %
    % Based on Corollary 3.3 from Chang, Choi & Park (2017)
    %% =======================================================================
    %% First case: |alpha| < 1 and |rho| < 1
    if abs(alpha) < 1 && abs(rho) < 1
        
        % Mean and variance from Corollary 3.3(a)
        mu_w = rho * uprec;
        sigma_w_sq = (1 - rho^2 + alpha^2 * rho^2) / (1 - alpha^2);
        sigma_w = sqrt(sigma_w_sq);
        
        % Base Gaussian density N(w; rho*u_{t-1}, sigma_w^2)
        base_dens = normpdf(w, mu_w, sigma_w);
        
        % Truncation argument
        z_arg = sqrt((1 - rho^2 + alpha^2 * rho^2) / (1 - rho^2)) * ...
                (tau - alpha * (w - rho * uprec) / (1 - rho^2 + alpha^2 * rho^2));
        
        % Normalization constant (denominator)
        norm_const = normcdf(tau * sqrt(1 - alpha^2));
        
        % Prec state = high state
        if sprec == 1
            truncation = 1 - normcdf(z_arg);
            pwcond = base_dens * truncation / (1 - norm_const);
        % Prec state = low state
        else
            truncation = normcdf(z_arg);
            pwcond = base_dens * truncation / norm_const;
        end
    
    %% 2nd case: |alpha| < 1 and |rho| = 1
    elseif abs(alpha) < 1 && abs(rho) == 1
        
        % Base density coefficient
        coeff = sqrt(1 - alpha^2) / abs(alpha);
        
        % Argument for density
        arg = (w - rho * uprec) / (alpha * sqrt(1 - alpha^2));
        
        % Base density (CORRECTED: normpdf not normcdf)
        base_dens = coeff * normpdf(arg);
        
        % Normalization
        norm_const = normcdf(tau * sqrt(1 - alpha^2));
        
        % Prec state = high state
        if sprec == 1
            indic = double(w > alpha * tau + rho * uprec);
            pwcond = base_dens * indic / (1 - norm_const);
        % Prec state = low state
        else
            indic = double(w <= alpha * tau + rho * uprec);
            pwcond = base_dens * indic / norm_const;
        end
    
    %% 3rd case: alpha = 1 and |rho| < 1
    elseif abs(alpha) == 1 && abs(rho) < 1
        
        % Mean and variance from Corollary 3.3(c)
        mu_w = rho * uprec;
        sigma_w_sq = (t - t * rho^2 + rho^2) / (t - 1);
        sigma_w = sqrt(sigma_w_sq);
        
        % Base Gaussian density (CORRECTED: normpdf(w, mu, sigma))
        base_dens = normpdf(w, mu_w, sigma_w);
        
        % Truncation argument
        z_arg = sqrt((t - t * rho^2 + rho^2) / (1 - rho^2)) * ...
                (tau - (w - rho * uprec) / (t - t * rho^2 + rho^2));
        
        % Normalization
        norm_const = normcdf(tau / sqrt(t - 1));
        
        % Prec state = high state
        if sprec == 1
            truncation = 1 - normcdf(z_arg);
            pwcond = base_dens * truncation / (1 - norm_const);
        % Prec state = low state
        else
            truncation = normcdf(z_arg);
            pwcond = base_dens * truncation / norm_const;
        end
    
    %% 4th case: alpha = 1 and |rho| = 1
    elseif abs(alpha) == 1 && abs(rho) == 1
        
        % Base density coefficient (CORRECTED: sqrt outside)
        coeff = sqrt(1 / (t - 1));
        
        % Argument for density
        arg = (w - rho * uprec) / sqrt(t - 1);
        
        % Base density
        base_dens = coeff * normpdf(arg);
        
        % Normalization
        norm_const = normcdf(tau / sqrt(t - 1));
        
        % Indicator function
        indic = double(w >= tau + rho * uprec);
        
        % Prec state = high state
        if sprec == 1
            pwcond = base_dens * indic / (1 - norm_const);
        % Prec state = low state
        else
            pwcond = base_dens * (1 - indic) / norm_const;
        end
        
    %% Error case
    else
        error('transition_density_latent: Invalid parameter combination: alpha=%.4f, rho=%.4f', alpha, rho);
    end
    
    end