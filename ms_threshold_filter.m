function LIKV = ms_threshold_filter(x)

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

% Vectors for the predicted probability and updated probability
predicted_density = zeros(T,1);
updating_density = zeros(T,1);

% Number of lags in the observed AR process
k = length(arcoefs);

% Number of regimes we are considering (max in case we considering change
% in vol but mean constant) -> pas besoin si mêmes coefs
nb_regimes = max(length(vecMu), length(vecSigma));

% Initialization of the likelihood (to check, log(p(y_1))
LIKV = 0;

% Vector or the state ordering (we compute in all state: lower ->
% higher)
s = 0:nb_regimes;

% Initialization of the transition probability for time K-1 (-> prec
% updating)

 % for loop (starting at t = k+2)
 for t = (k+2):T
     
     %% First part of the algorithm: Prediction step
     % computation of residuals at time t-1 within each regime
     resid_prec = (endog(t-1) - mu - arcoefs * endog(t-k-1:t-2))/sigma;

     % Computation of the transition probability (should return a vector)
     % -> à voir sur les dims
     transition_proba = transition_proba(s, alpha, rho, tau, resid_prec, t);
    
     % Computation of the transition density at time t
     transition_density = (1-s)*transition_proba + s * transition_proba;

     % Computation of the predicted densities (see for prec updating) -> A
     % revoir sur les sums, plus de l'archi pure --> Pas une sum des 4 (2 +
     % 2, mat 2,1)
     predicted_density = sum(transition_density);

     %% Second part of the algorithm: Computation of the likelihood
     % Computation of the conditional density of y given s
     condi_density_ys = normpdf(endog(t), mu+phi*endog(t-k:t-1),sigma).*predicted_density;
     
     % Sum over the regimes and likelihood computation
     condi_density_y = sum(condi_density_ys);
     LL = -1 * log(condi_density_y);

     %% Third part of the algorithm: Updating step
     updating_density(t,1) = (condi_density_ys.*predicted_density)/condi_density_y;
     
     %% Fourth part: aggregating the likelihood
     LIKV = LIKV + LL;
 end

end

