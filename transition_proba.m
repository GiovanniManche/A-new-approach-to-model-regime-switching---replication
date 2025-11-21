function [omega] = transition_proba(sprec, wprec, alpha, rho, tau, u, t)

% Function to compute the transition probability for a 
% 2 states MS-Threshold Model 
% 
% Inputs: 
%
% - sprec: regime of the precedent period
% - wprec: precedent value for the latent process
% - alpha: autoregressive coefficient of the latent process
% - rho: correlation coefficient between residuals of latent and observed
% process
% - tau: threshold 
% - u: precedent value of the residuals of observed process
% - t: prece

% Output: 
%
% omega: transition probability to the low state conditional on the
% previous states and past values of observed time series

% Useful elements for computation
absrho = abs(rho);
num_elem = normcdf((tau - rho*u)*sqrt(1-alpha^2)/alpha, 0, 1);
denom_elem = normcdf(tau*sqrt(1-alpha^2), 0, 1);
x = wprec * sqrt(1 - alpha^2);

% Function integrated to compute transition probabilities (--> à revoir :
% Phi_p peut être une N bivariée, attention)
fun_statio = @(x, alpha, tau, u, rho) normcdf(((tau - rho * u - alpha * x / (sqrt(1-alpha^2)))/sqrt(1-rho^2)), 0, 1)...
    * normpdf(x);

fun_rw = @(x, t, tau, u, rho) normcdf(((tau - rho * u - x * sqrt(t-1)) /sqrt(1-rho^2)), 0, 1)...
    * normpdf(x);

% First case: |rho| = 1 and alpha = 0
if absrho == 1 && alpha == 0
    omega = rho * u < tau;
   
% Second case: |rho| = 1 and 0 < alpha < 1
elseif absrho == 1 && (0 < alpha && alpha < 1)
    omega = (1-sprec) * min(1, num_elem/denom_elem) ...
        + sprec * max(0, (num_elem - denom_elem)/(1-denom_elem));

% Third case: |rho| = 1 and -1 < alpha < 0
elseif absrho == 1 && (-1 < alpha && alpha < 0)
    omega = sprec * min(1, (1-num_elem)/(1-denom_elem)) + (1-sprec) * ...
        max((denom_elem - num_elem)/denom_elem);

% Fourth case: |rho| = 1 and alpha = 1
elseif absrho == 1 && alpha == 1
    % Case where t=1
    if t == 1
        omega = normcdf(tau - rho * u, 0, 1);
     
    else
        % Case rho * u > 0
        if rho * u > 0
            omega = 1 - sprec;
        else
            omega = (normcdf((tau - rho * u)/sqrt(t-1), 0, 1)-sprec * normcdf(tau/sqrt(t-1), 0, 1))...
                /((1-sprec)*normcdf(tau/sqrt(t-1), 0, 1) + sprec * (1 - normcdf(tau/sqrt(t-1), 0, 1)));
        end
    end

% Other cases of rho:
% First case: alpha = 1 and |rho| < 1
elseif alpha == 1 && absrho < 1

    % Case for t = 1
    if t == 1
        omega = normcdf(tau, 0, 1);
    else
        bound = tau / sqrt(tau - 1);
        denom = (1-sprec) * normcdf(bound, 0, 1) + sprec(1-normcdf(bound);
        omega = ((1-sprec) * integral(@(x) fun_rw(x, t, tau, u, rho), -Inf, bound) + sprec * integral(@(x) fun_rw(x, t, tau, u, rho), bound, Inf))...
            / denom;
    end

% Case |alpha| < 1 and |rho| < 1
elseif abs(alpha) < 1 && absrho < 1
    bound = tau * sqrt(1-alpha^2);
    denom = (1-sprec) * normcdf(bound, 0, 1) + sprec(1 - normcdf(bound, 0, 1));
    omega = ((1-sprec) * integral(@(x) fun_statio(x, alpha, tau, u, rho), -Inf, bound) + ...
        sprec * integral(@(x) fun_statio(x, alpha, tau, u, rho), bound, Inf))...
            / denom;

% Other cases : not supposed to happen
else
    disp("Alpha value :");
    disp(alpha);
    disp("Rho value:");
    disp(rho);
    error("Error in transition proba computation");
end
end

