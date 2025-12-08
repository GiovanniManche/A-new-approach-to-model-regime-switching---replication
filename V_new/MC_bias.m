%% 1. CONFIGURATION
clearvars -except Results; clc; close all;

% Output files
text_filename = 'MonteCarlo_Bias_Report.txt';
data_filename = 'MC_Raw_Data.mat';

fid = fopen(text_filename, 'w');
if fid == -1, error('Cannot open %s', text_filename); end

% --- SIMULATION RANGE ---
% IMPORTANT: Change to 1:500 or 1:1000 for the final paper
% For testing tonight, maybe use 1:50 or 1:100
sim_indices = 1:100; 

% --- PARAMETERS & SCENARIOS ---
% Structure: {Index in Results, True Value, Label}
rho_scenarios = {
    1,   0.0,  'Rho_0';
    6,  -0.5,  'Rho_Mid';
    11, -1.0,  'Rho_Endog'
};

% Mean Model Base Params (k=1)
% Order: [Mu_L, Mu_H, Sigma, Alpha, Tau, Rho, Gamma]
base_mean = [0.6; 3.0; 0.8; 0.4; 0.5; NaN; 0.5]; 
names_mean = {'Mu_L', 'Mu_H', 'Sigma', 'Alpha', 'Tau', 'Rho', 'Gamma'};

% Vol Model Base Params (k=0)
% Order: [Mu, Sig_L, Sig_H, Alpha, Tau, Rho]
base_vol  = [0.0; 0.04; 0.12; 0.4; 0.5; NaN];
names_vol = {'Mu', 'Sig_L', 'Sig_H', 'Alpha', 'Tau', 'Rho'};

% Storage structure for the visualization script
MC_Data = struct();

%% 2. MAIN LOOP
models_list = {'Mean', 'Vol'};

for m = 1:length(models_list)
    model_type = models_list{m};
    is_mean = strcmp(model_type, 'Mean');
    
    if is_mean
        k_lag = 1; run_type = 'mean';
        base_true_params = base_mean; param_names = names_mean; rho_pos = 6;
    else
        k_lag = 0; run_type = 'volatility'; 
        base_true_params = base_vol; param_names = names_vol; rho_pos = 6;
    end
    
    num_params = length(base_true_params);
    num_sims = length(sim_indices);
    
    dual_print(fid, '\n############################################################\n');
    dual_print(fid, ' MONTE CARLO ANALYSIS: %s MODEL (N=%d)\n', upper(model_type), num_sims);
    dual_print(fid, '############################################################\n');

    % LOOP OVER RHO SCENARIOS
    for r = 1:size(rho_scenarios, 1)
        curr_rho_idx = rho_scenarios{r, 1};
        curr_rho_val = rho_scenarios{r, 2};
        scenario_lbl = rho_scenarios{r, 3};
        
        % Set true parameter vector for this scenario
        true_params = base_true_params;
        true_params(rho_pos) = curr_rho_val;
        
        % Storage matrices for this specific scenario
        estimates_raw = nan(num_params, num_sims);
        se_raw        = nan(num_params, num_sims);
        
        dual_print(fid, '\n---> SCENARIO: %s (True Rho = %.1f)\n', scenario_lbl, curr_rho_val);
        
        % --- SIMULATION LOOP ---
        % Tip: Use 'parfor' instead of 'for' if you have the Parallel Toolbox
        for s_idx = 1:num_sims
            curr_sim = sim_indices(s_idx);
            
            % 1. Extract Data
            if is_mean
                y = Results.Pair_1.Mean(:, curr_sim, curr_rho_idx);
            else
                y = Results.Pair_1.Vol(:, curr_sim, curr_rho_idx);
            end
            
            % 2. Run Estimation
            % 4th arg = false (Skip Exogenous Test to save time)
            if mod(s_idx, 10) == 0
                fprintf('Processing %s | Rho %.1f | Sim %d/%d...\n', model_type, curr_rho_val, s_idx, num_sims);
            end
            
            try
                % Suppress output for speed
                [~, output] = evalc('run_full_analysis(y, k_lag, run_type, false, false);');
                
                estimates_raw(:, s_idx) = output.params_estimated;
                se_raw(:, s_idx)        = output.se;
            catch
                fprintf('  [Warning] Optimization failed for Sim %d. Skipping.\n', curr_sim);
            end
        end
        
        % --- COMPUTE STATISTICS ---
        % Filter out NaNs (failed runs)
        valid_mask = ~any(isnan(estimates_raw), 1);
        clean_est = estimates_raw(:, valid_mask);
        clean_se  = se_raw(:, valid_mask);
        n_valid = sum(valid_mask);
        
        % 1. Mean Estimate
        mean_est = mean(clean_est, 2);
        
        % 2. BIAS = Mean(Estimate) - True
        bias = mean_est - true_params;
        
        % 3. RMSE = Sqrt( Mean( (Estimate - True)^2 ) )
        mse = mean((clean_est - true_params).^2, 2);
        rmse = sqrt(mse);
        
        % 4. SD (Monte Carlo Standard Deviation)
        std_dev = std(clean_est, 0, 2);
        
        % --- SAVE DATA FOR PLOTTING ---
        % We save everything in a structured way
        field_name = sprintf('%s_%s', model_type, scenario_lbl); % e.g., Mean_Rho_0
        MC_Data.(field_name).estimates = clean_est;
        MC_Data.(field_name).se = clean_se;
        MC_Data.(field_name).true_rho = curr_rho_val;
        MC_Data.(field_name).bias = bias(rho_pos);
        MC_Data.(field_name).rmse = rmse(rho_pos);
        
        % --- DISPLAY TABLE ---
        display_stats_table(fid, param_names, true_params, mean_est, bias, rmse, std_dev, n_valid);
    end
end

% SAVE EVERYTHING FOR THE PLOTTING SCRIPT
save(data_filename, 'MC_Data');
fclose(fid);
fprintf('\nDone! Results saved in %s and data in %s\n', text_filename, data_filename);

%% HELPER FUNCTIONS
function display_stats_table(fid, names, true_p, mean_p, bias, rmse, std_dev, N)
    dual_print(fid, '\n  AGGREGATE RESULTS (N = %d valid simulations)\n', N);
    dual_print(fid, '  --------------------------------------------------------------------------------------\n');
    dual_print(fid, '  %-10s | %8s | %8s | %8s | %8s | %8s\n', 'Param', 'TRUE', 'MEAN EST', 'BIAS', 'RMSE', 'SD');
    dual_print(fid, '  --------------------------------------------------------------------------------------\n');
    
    for i = 1:length(names)
        dual_print(fid, '  %-10s | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n', ...
            names{i}, true_p(i), mean_p(i), bias(i), rmse(i), std_dev(i));
    end
    dual_print(fid, '  --------------------------------------------------------------------------------------\n');
end

function dual_print(fid, varargin)
    fprintf(varargin{:});
    if fid > 0
        fprintf(fid, varargin{:});
    end
end