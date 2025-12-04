function params = create_params(alpha, tau, rho, mu, gamma, sigma, k)
%==========================================================================
% CREATE_PARAMS  Build a parameter structure for an r-regime AR(k) model.
%
% Inputs:
%   - alpha : AR coefficient of the latent factor (|alpha| <= 1)
%   - tau   : threshold on the latent factor
%   - rho   : correlation between u_t and v_{t+1} (endogeneity), in [-1,1]
%   - mu    : (r x 1) vector of regime means   [mu(1), ..., mu(r)]'
%   - gamma : scalar or (k x 1) vector of AR coefficients for y_t
%   - sigma : (r x 1) vector of regime volatilities (sigma > 0)
%   - k     : AR order on y_t (integer >= 0)
%
% Output:
%   - params: structure with fields:
%       alpha, tau, rho, mu (r x 1), gamma (k x 1), sigma (r x 1),
%       k (AR order), r (number of regimes)
%==========================================================================

    %% Basic structure 
    params = struct();
    params.alpha = alpha;
    params.tau   = tau;
    params.rho   = rho;

    % Ensure column vectors
    params.mu    = mu(:);
    params.sigma = sigma(:);

    params.gamma = gamma(:);  % can be scalar or vector
    params.k     = k;
    params.r     = length(params.mu);

    %% Checks 

    % 1. |alpha| <= 1
    if abs(params.alpha) > 1
        error('alpha must be in [-1,1], got %.4f', params.alpha);
    end

    % 2. rho in [-1,1]
    if abs(params.rho) > 1
        error('rho must be in [-1,1], got %.4f', params.rho);
    end

    % 3. sigma > 0
    if any(params.sigma <= 0)
        error('All entries of sigma must be > 0.');
    end

    % 4. k must be integer >= 0 (allow k = 0 = no AR)
    if params.k < 0 || params.k ~= floor(params.k)
        error('k must be an integer >= 0, got %.4f', params.k);
    end

    % 5. gamma length vs k
    if params.k == 0
        % no AR, so gamma can be empty or scalar 0
        if ~isempty(params.gamma) && any(params.gamma ~= 0)
            warning('k=0 but gamma is non-zero; gamma will be ignored in the model.');
        end
    else
        if length(params.gamma) ~= params.k
            if params.k == 1 && isscalar(params.gamma)
                % OK: scalar gamma for AR(1)
            else
                error('gamma must have length k=%d, got length %d.', ...
                      params.k, length(params.gamma));
            end
        end
    end

    % 6. mu and sigma must have same length
    if length(params.mu) ~= length(params.sigma)
        error('mu and sigma must have the same length (number of regimes).');
    end
end
