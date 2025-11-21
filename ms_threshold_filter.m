function [LL] = ms_threshold_filter(x)

% Function which computes the likelihood for a MS model with threshold
% Inputs:
% 
% Vector of parameters x containing: 
% - mu: vector containing the average of the observed process in both
% states
% - sigma: vector containing the volatility of the observed process in both
% states
% - tau: threshold level
% - alpha: autoregressive coefficient for the latent process
% - rho: correlation coefficient between residuals of latent and observed
% process
% - arcoefs: vector containing the p coefficients of the observed AR
% process

% Affectation of the serie 
global endog;

% Affectation of all parameters
mu = x(1); sigma = x(2);
tau = x(3); 
alpha = x(4);
rho = x(5);
arcoefs = x(6);

% Length of the serie
T = length(endog);

% Number of lags in the observed AR process
k = length(arcoefs);

% Initialization of the likelihood (to check, log(p(y_1))
LIKV = 0;

% Vectors for the latent process and the state vector
w = zeros(T,1);
s = w; 

% Initialization of w depending on alpha coef
if abs(alpha) == 1
    w(k,1) = 0;
elseif abs(alpha) < 1
    w(k,1) = normrnd(0, 1/(1-alpha^2));
else
    error("Alpha shouldn't be superior to 1 in absolute value");
end

% Associated state
s(k,1) = w > tau;


 % for loop (starting at t = k+1)

end

