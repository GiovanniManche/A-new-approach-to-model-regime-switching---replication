function [results] = optimize_model(y, k, model_type, options)
%% FUNCTION DESCRIPTION
% Estimates Regime Switching (Mean OR Volatility)
%   INPUTS:
%     y          : Time series (T x 1)
%     k          : AR order of the model (not the latent factor)
%     model_type : 'mean' (2 mu, 1 sigma) OR 'volatility' (1 mu, 2 sigma)
%     options    : struct with .fix_rho (optional)
%   OUTPUT:
%     results    : structure of parameters, with their optimised values

    if nargin < 4, options = struct(); end
    y = y(:); % Force column vector
    
    is_volatility = strcmpi(model_type, 'volatility');

    % 1. Initial Guess (OLS) 
    mu_init    = mean(y);
    sigma_init = std(y);
    gamma_init = zeros(1, k);

    if k > 0
        T = length(y);
        Y_vec = y(k+1:T);
        X_mat = zeros(T-k, k);
        for j = 1:k, X_mat(:, j) = y(k+1-j:T-j); end
        phi = (X_mat' * X_mat) \ (X_mat' * Y_vec); 
        gamma_init = phi.'; 
    end

    alpha_init = 0.5;
    tau_init   = 0.0;
    rho_init   = -0.5;

    fix_rho_val = [];
    if isfield(options, 'fix_rho')
        fix_rho_val = options.fix_rho;
        rho_init    = fix_rho_val;
    end

    % 2. Construct Parameters & Constraints 
    % Common Indices for Tail: Alpha=4, Tau=5, Rho=6, Gamma=7...
    
    if is_volatility
        % === VOLATILITY MODEL ===
        % Structure: [mu, sigma_low, sigma_high, alpha, tau, rho, gamma...]
        % Constraint: 1 mu, 2 sigmas. We force sigma_low < sigma_high.
        
        x0 = [ ...
            mu_init, ...                % 1. mu
            sigma_init * 0.5, ...       % 2. sigma_low
            sigma_init * 1.5, ...       % 3. sigma_high
            alpha_init, ...             % 4. alpha
            tau_init, ...               % 5. tau
            rho_init, ...               % 6. rho
            gamma_init ...              % 7+. gamma
        ];
        
        lb = [-inf, 1e-4, 1e-4, -0.99, -inf, -0.99, -inf(1, k)];
        ub = [inf, inf, inf, 0.99, inf, 0.99, inf(1, k)];
        
        % Linear Constraint: sigma_low < sigma_high
        % x(2) - x(3) <= -1e-4
        A = zeros(1, length(x0));
        A(2) = 1; A(3) = -1;
        b = -1e-4;

    else
        % === MEAN MODEL ===
        % Structure: [mu_low, mu_high, sigma, alpha, tau, rho, gamma...]
        % Constraint: 2 mus, 1 sigma. We force mu_low < mu_high.
        
        x0 = [ ...
            mu_init - 0.5*sigma_init, ... % 1. mu_low
            mu_init + 0.5*sigma_init, ... % 2. mu_high
            sigma_init, ...               % 3. sigma 
            alpha_init, ...               % 4. alpha
            tau_init, ...                 % 5. tau
            rho_init, ...                 % 6. rho
            gamma_init ...                % 7+. gamma
        ];

        lb = [-inf, -inf, 1e-4, -0.99, -inf, -0.99, -inf(1, k)];
        ub = [inf, inf, inf, 0.99, inf, 0.99, inf(1, k)];
        
        % Linear Constraint: mu_low < mu_high
        % x(1) - x(2) <= -1e-4
        A = zeros(1, length(x0));
        A(1) = 1; A(2) = -1;
        b = -1e-4;
    end

    % Handle Fixed Rho (Index 6 in both cases)
    idx_rho = 6;
    if ~isempty(fix_rho_val)
        x0(idx_rho) = fix_rho_val;
        lb(idx_rho) = fix_rho_val;
        ub(idx_rho) = fix_rho_val;
    end

    % 3. Optimization 
    opt_fmincon = optimoptions('fmincon', ...
        'Algorithm', 'sqp', ...
        'Display', 'iter', ...
        'MaxFunctionEvaluations', 10000, ...
        'MaxIterations', 1000, ...
        'StepTolerance', 1e-6, ...
        'OptimalityTolerance', 1e-5);

    % Objective Wrapper
    obj_fun = @(p) likelihood_function(p, y, k, is_volatility);

    fprintf('Starting Optimization for %s model...\n', upper(model_type));

    try
        [params_hat, fval, exitflag, output, ~, ~, hessian] = ...
            fmincon(obj_fun, x0, A, b, [], [], lb, ub, [], opt_fmincon);

        results.params   = params_hat;
        results.loglik   = -fval;
        results.exitflag = exitflag;
        results.output   = output;
        
        % Filtered probabilities
        [~, filtered_probs] = likelihood_function(params_hat, y, k, is_volatility);
        results.filtered_probs = filtered_probs;

        % Robust standard errors
        if rcond(hessian) < 1e-12
            cov_matrix = pinv(hessian);
        else
            cov_matrix = inv(hessian);
        end
        se = sqrt(diag(cov_matrix))';
        if ~isreal(se), se = real(se); end
        results.se = se;
        
        fprintf('Optimization Finished. Exit Flag: %d\n', exitflag);

    catch ME
        fprintf('\nCRITICAL ERROR in optimization: %s\n', ME.message);
        results.params = []; results.loglik = -Inf;
        rethrow(ME);
    end
end