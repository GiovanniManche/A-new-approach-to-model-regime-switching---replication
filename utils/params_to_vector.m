function theta_vec = params_to_vector(params)
%% FUNCTION DESCRIPTION
% Convert parameter structure into a column vector for optimisation.
%
% Input:
%   params: structure with fields
%       alpha, tau, rho,
%       mu    (r x 1),
%       gamma (k x 1),
%       sigma (r x 1),
%       k, r
%
% Output:
%   theta_vec: column vector
%     theta_vec = [alpha; tau; rho; mu(:); gamma(:); sigma(:)]
%% =======================================================================

    %% Validation 
    if ~isstruct(params)
        error('Input must be a struct.');
    end

    required_fields = {'alpha', 'tau', 'rho', 'mu', 'gamma', 'sigma', 'k', 'r'};
    for i = 1:length(required_fields)
        if ~isfield(params, required_fields{i})
            error('Missing field in params: %s', required_fields{i});
        end
    end

    %% Extract and ensure column vectors
    alpha = params.alpha;
    tau   = params.tau;
    rho   = params.rho;

    mu    = params.mu(:);
    gamma = params.gamma(:);
    sigma = params.sigma(:);

    %% Basic consistency checks
    r = params.r;
    if length(mu) ~= r || length(sigma) ~= r
        error('Length of mu and sigma must both equal r = %d.', r);
    end

    k = params.k;
    if (k > 0 && length(gamma) ~= k)
        error('Length of gamma must equal k = %d.', k);
    end

    %% Construct vector 
    % ORDER IS IMPORTANT and must match vector_to_params()
    theta_vec = [
        alpha;
        tau;
        rho;
        mu;
        gamma;
        sigma
    ];
end
