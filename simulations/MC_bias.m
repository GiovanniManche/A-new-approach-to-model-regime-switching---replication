function MC_Data = MC_bias(Results, sim_indices, output_prefix)
    %% FUNCTION DESCRIPTION
    % Runs the full Monte Carlo analysis (Mean & Volatility models) across
    % different scenarios of Rho.
    %
    % INPUTS:
    %   Results       : Structure containing simulation results (Results.Pair_1...)
    %   sim_indices   : (Optional) Vector of indices (e.g. 1:100). Default: 1:50
    %                   For full analysis: 1:1000
    %   output_prefix : (Optional) String prefix for filenames. Default: 'MC_Results'
    %
    % OUTPUTS:
    %   MC_Data       : Structure containing all estimates, bias, RMSE, etc.
    %                   (Also saved to .mat file)
    %% ====================================================================

    %% 1. CONFIGURATION
    if nargin < 2 || isempty(sim_indices)
        sim_indices = 1:50; 
    end
    if nargin < 3 || isempty(output_prefix)
        output_prefix = 'MC_Results';
    end

    % Filenames
    text_filename = [output_prefix, '.txt'];
    data_filename = [output_prefix, '.mat'];

    % Open Text File
    fid = fopen(text_filename, 'w');
    if fid == -1, error('Cannot open %s', text_filename); end
    % Ensure file closes even if crash occurs
    cleanupObj = onCleanup(@() fclose(fid));

    % Structure and scenarios
    rho_scenarios = {
        1,   0.0,  'Rho_0';
        6,  -0.5,  'Rho_Mid';
        11, -1.0,  'Rho_Endog'
    };
    
    % For the base parameters, we use those used in the main
    % Mean Model Base Params (k=1)
    base_mean = [0.6; 3.0; 0.8; 0.4; 0.5; NaN; 0.5]; 
    names_mean = {'Mu_L', 'Mu_H', 'Sigma', 'Alpha', 'Tau', 'Rho', 'Gamma'};

    % Vol Model Base Params (k=0)
    base_vol  = [0.0; 0.04; 0.12; 0.4; 0.5; NaN];
    names_vol = {'Mu', 'Sig_L', 'Sig_H', 'Alpha', 'Tau', 'Rho'};

    % Storage structure
    MC_Data = struct();

    %% 2. MAIN LOOP
    models_list = {'Mean', 'Vol'};
    
    for m = 1:length(models_list)
        model_type = models_list{m};
        is_mean = strcmp(model_type, 'Mean');
        
        % Model specific setup
        if is_mean
            k_lag = 1; run_type = 'mean';
            base_true_params = base_mean; param_names = names_mean; rho_pos = 6;
        else
            k_lag = 0; run_type = 'volatility'; 
            base_true_params = base_vol; param_names = names_vol; rho_pos = 6;
        end
        
        num_params = length(base_true_params);
        num_sims = length(sim_indices);
        % Log for user + print in the text file
        dual_print(fid, '\n############################################################\n');
        dual_print(fid, ' MONTE CARLO ANALYSIS: %s MODEL (N=%d)\n', upper(model_type), num_sims);
        dual_print(fid, '############################################################\n');

        % LOOP OVER RHOS
        for r = 1:size(rho_scenarios, 1)
            curr_rho_idx = rho_scenarios{r, 1};
            curr_rho_val = rho_scenarios{r, 2};
            scenario_lbl = rho_scenarios{r, 3};
            
            true_params = base_true_params;
            true_params(rho_pos) = curr_rho_val;
            
            % Storage matrices 
            estimates_raw = nan(num_params, num_sims);
            se_raw        = nan(num_params, num_sims);
            
            dual_print(fid, '\n---> SCENARIO: %s (True Rho = %.1f)\n', scenario_lbl, curr_rho_val);
            
            % LOOP ON THE SIMULATIONS
            for s_idx = 1:num_sims
                curr_sim = sim_indices(s_idx);
                
                % 1. Extract Data
                if is_mean
                    % Check bounds
                    if curr_sim > size(Results.Pair_1.Mean, 2)
                        warning('Sim %d out of bounds for Mean data', curr_sim); continue;
                    end
                    y = Results.Pair_1.Mean(:, curr_sim, curr_rho_idx);
                else
                    if curr_sim > size(Results.Pair_1.Vol, 2)
                        warning('Sim %d out of bounds for Vol data', curr_sim); continue;
                    end
                    y = Results.Pair_1.Vol(:, curr_sim, curr_rho_idx);
                end
                
                % Progression message
                if mod(s_idx, 50) == 0 || s_idx == 1
                    fprintf('Processing %s | Rho %.1f | Sim %d/%d...\n', model_type, curr_rho_val, s_idx, num_sims);
                end
                
                % 2. Run Estimation
                try
                    % We use evalc to suppress the command window output of run_full_analysis
                    % This keeps the console clean during 1000 simulations
                    % Also we do not plot results (1000 windows...)
                    [~, output] = evalc('run_full_analysis(y, k_lag, run_type, false, false);');
                    
                    estimates_raw(:, s_idx) = output.params_estimated;
                    se_raw(:, s_idx)        = output.se;
                catch ME
                    fprintf('  [Warning] Optimization failed for Sim %d: %s\n', curr_sim, ME.message);
                end
            end
            
            % COMPUTE STATISTICS
            % Filter out NaNs (failed runs)
            valid_mask = ~any(isnan(estimates_raw), 1);
            clean_est = estimates_raw(:, valid_mask);
            clean_se  = se_raw(:, valid_mask);
            n_valid = sum(valid_mask);
            
            if n_valid == 0
                warning('No valid simulations for %s %s', model_type, scenario_lbl);
                continue;
            end
            
            % 1. Mean Estimate
            mean_est = mean(clean_est, 2);
            
            % 2. BIAS = Mean(Estimate) - True
            bias = mean_est - true_params;
            
            % 3. RMSE = Sqrt( Mean( (Estimate - True)^2 ) )
            mse = mean((clean_est - true_params).^2, 2);
            rmse = sqrt(mse);
            
            % 4. SD (Monte Carlo Standard Deviation)
            std_dev = std(clean_est, 0, 2);
            
            % Save results
            field_name = sprintf('%s_%s', model_type, scenario_lbl); 
            MC_Data.(field_name).estimates = clean_est;
            MC_Data.(field_name).se = clean_se;
            MC_Data.(field_name).true_rho = curr_rho_val;
            MC_Data.(field_name).bias = bias(rho_pos);
            MC_Data.(field_name).rmse = rmse(rho_pos);
            MC_Data.(field_name).full_bias_vector = bias; 
            MC_Data.(field_name).full_rmse_vector = rmse; 
            
            % Display table
            display_stats_table(fid, param_names, true_params, mean_est, bias, rmse, std_dev, n_valid);
        end
    end

    % Save all
    save(data_filename, 'MC_Data');
    fprintf('\nDone! Results saved in %s and %s\n', text_filename, data_filename);
end

%% HELPER FUNCTIONS
function display_stats_table(fid, names, true_p, mean_p, bias, rmse, std_dev, N)
% Function that print the coefficients in a standardized table: 
% Name of the parameter | known true value | estimated value (average 
%       over all the simulations) | Bias | RMSE | standard error
    dual_print(fid, '\n  AGGREGATE RESULTS (N = %d valid simulations)\n', N);
    dual_print(fid, '  %s\n', repmat('-', 1, 80));
    dual_print(fid, '  %-10s | %8s | %8s | %8s | %8s | %8s\n', 'Param', 'TRUE', 'MEAN EST', 'BIAS', 'RMSE', 'SD');
    dual_print(fid, '  %s\n', repmat('-', 1, 80));
    
    for i = 1:length(names)
        dual_print(fid, '  %-10s | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n', ...
            names{i}, true_p(i), mean_p(i), bias(i), rmse(i), std_dev(i));
    end
    dual_print(fid, '  %s\n', repmat('-', 1, 80));
end

function dual_print(fid, varargin)
    % Function that allows to both display the results in MATLAB and save
    % them in a text file
    fprintf(varargin{:});
    if fid > 0
        fprintf(fid, varargin{:});
    end
end