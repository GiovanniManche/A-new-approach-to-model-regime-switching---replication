function Results = run_monte_carlo_simulations(n_obs, n_simul, rho_values,...
    alpha_tau_pairs, base_vol_params, base_mean_params)
%% FUNCTION DESCRIPTION
% Runs Monte Carlo simulations for Regime Switching models
%
% INPUTS:
%   n_obs           : Number of observations per simulation (e.g., 500)
%   n_simul         : Number of simulations per configuration (e.g., 1000)
%   rho_values      : Vector of rho values to test (e.g., 0:-0.1:-1)
%   alpha_tau_pairs : Matrix (Nx2) of [alpha, tau] pairs
%   base_vol_params : (Optional) Base params structure for Volatility Model
%   base_mean_params: (Optional) Base params structure for Mean Model
%
% OUTPUT:
%   Results         : Struct containing simulated data (Vol and Mean)

    %% 1. Handle Optional Inputs (Defaults)
    if nargin < 5 || isempty(base_vol_params)
        % Default Volatility Params (from your script)
        % mu=0, sigma=[0.04; 0.12], gamma=0
        base_vol_params = create_params(0, 0, 0, 0, [0.04;0.12], 0);
    end
    
    if nargin < 6 || isempty(base_mean_params)
        % Default Mean Params (from your script)
        % mu=[0.6; 3], sigma=0.8, gamma=0.5
        base_mean_params = create_params(0, 0, 0, [0.6;3], 0.8, 0.5);
    end

    %% 2. Initialization
    n_rho = length(rho_values);
    n_pairs = size(alpha_tau_pairs, 1);
    Results = struct();
    
    fprintf('Starting Monte Carlo: %d Pairs, %d Rho values, %d Sims each.\n', ...
        n_pairs, n_rho, n_simul);

    %% 3. Monte Carlo Loops
    
    % Loop 1: Pairs of Alpha/Tau
    for p = 1:n_pairs
        curr_alpha = alpha_tau_pairs(p, 1);
        curr_tau   = alpha_tau_pairs(p, 2);
        
        fprintf('  > Processing Pair %d/%d (Alpha=%.2f, Tau=%.2f)...\n', ...
            p, n_pairs, curr_alpha, curr_tau);

        % Pre-allocate 3D matrices
        % Dims: (Observations x Simulations x Rho_Values)
        storage_vol  = zeros(n_obs, n_simul, n_rho);
        storage_mean = zeros(n_obs, n_simul, n_rho);

        % Update Alpha/Tau
        temp_vol_params = base_vol_params;
        temp_vol_params.alpha = curr_alpha;
        temp_vol_params.tau   = curr_tau;
        
        temp_mean_params = base_mean_params;
        temp_mean_params.alpha = curr_alpha;
        temp_mean_params.tau   = curr_tau;

        % Loop 2: Rho Values
        for r = 1:n_rho
            curr_rho = rho_values(r);
            
            % Update Rho
            temp_vol_params.rho  = curr_rho;
            temp_mean_params.rho = curr_rho;

            % Loop 3: Simulations
            for n = 1:n_simul
                % Simulate Volatility Model
                [y_vol, ~, ~, ~, ~] = simulate_regime_switching(n_obs, temp_vol_params, 'volatility');
                
                % Simulate Mean Model
                [y_mean, ~, ~, ~, ~] = simulate_regime_switching(n_obs, temp_mean_params, 'mean');
                
                % Store results
                storage_vol(:, n, r)  = y_vol;
                storage_mean(:, n, r) = y_mean;
            end
        end

        % Save to structure dynamically
        field_name = sprintf('Pair_%d', p);
        Results.(field_name).Vol   = storage_vol;
        Results.(field_name).Mean  = storage_mean;
        Results.(field_name).Alpha = curr_alpha;
        Results.(field_name).Tau   = curr_tau;
        Results.(field_name).Rhos  = rho_values; 
    end
    
    fprintf('Monte Carlo simulations complete!\n');
end