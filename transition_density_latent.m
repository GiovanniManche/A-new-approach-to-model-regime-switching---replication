function [pwcond] = transition_density_latent(w, uprec, sprec, tau, alpha, rho, t)
% Function to compute the transition density of the latent process
% 
% Inputs :
% - w: current value for the latenr variable
% - uprec: precedent value for the residual of endogenous process
% - sprec: precedent state (1 for high state, 0 for low state)
% - tau: threshold
% - alpha: autoregressive coefficient for the latent process
% - rho: correlation coefficient
% - t: time

% Output: p(wt|s_{t-1}, ..., s_{t-k-1}, F_{t-1})

% First case: |alpha| < 1 and |rho| < 1
if abs(alpha) < 1 && abs(rho) < 1

    % prec state = high state
    if sprec == 1
        pwcond = (1 - normcdf(sqrt(1-rho^2 + alpha^2 * rho^2)/(1-rho^2) * (tau - alpha(w - rho * uprec)/(1-rho^2+alpha^2*rho^2)), 0, 1))...
            /(1-normcdf(tau * sqrt(1-alpha^2), 0, 1)) ...
            * normrnd(rho * uprec, (1-rho^2 + alpha^2 * rho^2)/1-alpha^2);
    % prec state = low state
    else
        pwcond = (normcdf(sqrt(1-rho^2 + alpha^2 * rho^2)/(1-rho^2) * (tau - alpha(w - rho * uprec)/(1-rho^2+alpha^2*rho^2)), 0, 1)...
            /normcdf(tau * sqrt(1-alpha^2), 0, 1)) * normrnd(rho * uprec, (1-rho^2 + alpha^2 * rho^2)/1-alpha^2);
    end

% 2nd case: |alpha| < 1 and |rho| = 1
elseif abs(alpha) < 1 && abs(rho) == 1

    % Prec state = high state
    if sprec == 1
        indic = w > alpha * tau + rho * uprec;
        pwcond = (sqrt(1-alpha^2)/alpha * normcdf((w - rho * uprec)/alpha * sqrt(1-alpha^2), 0, 1))...
            /(1-normcdf(tau * sqrt(1-alpha^2), 0, 1)) * indic;
    % Prec state = low state
    else
        indic = w <= alpha * tau + rho * uprec;
        pwcond = (sqrt(1-alpha^2)/alpha * normcdf((w - rho * uprec)/alpha * sqrt(1-alpha^2), 0, 1))...
            /normcdf(tau * sqrt(1-alpha^2), 0, 1) * indic;
    end

% 3rd case: alpha = 1 and |rho| < 1
elseif abs(alpha) == 1 && abs(rho) < 1
    
    % mean and variance of Normal
    mu = rho * uprec;
    sigma = (t-t*rho^2+rho^2)/(t-1);

    % Prec state = high state
    if sprec == 1
        pwcond = (1-normcdf(sqrt((t - t * rho^2 + rho^2)/(1-rho^2))*(tau - (w - rho * uprec)/(t - t * rho^2 + rho^2)),0,1))...
            /(1-normcdf(tau/sqrt(t-1), 0, 1)) * normrnd(mu, sigma);
    
    % Prec state = low state
    else
        pwcond = normcdf(sqrt((t - t * rho^2 + rho^2)/(1-rho^2))*(tau - (w - rho * uprec)/(t - t * rho^2 + rho^2)),0,1)...
            /normcdf(tau/sqrt(t-1), 0, 1) * normrnd(mu, sigma);
    end

% 4th case: alpha = 1 and |rho| = 1
elseif abs(alpha) == 1 && abs(rho) == 1

    % Calculation of indicatric
    indic = w >= tau + rho * uprec;

    % prec state = High state
    if sprec == 1
        pwcond = (1/(sqrt(t-1) * normpdf((w - rho * uprec)/(t-1))))...
            /(1-normcdf(tau/sqrt(t-1), 0, 1)) * indic;
    % prec state = low state
    else:
        pwcond = (1/(sqrt(t-1) * normpdf((w - rho * uprec)/(t-1))))...
            /(1-normcdf(tau/sqrt(t-1), 0, 1)) * (1-indic);
    end
    
% else: error
else
    disp("Alpha value:");
    disp(alpha);
    disp("rho value");
    disp(rho);
    error("Either alpha or rho have an outer bounds value")
end
end

