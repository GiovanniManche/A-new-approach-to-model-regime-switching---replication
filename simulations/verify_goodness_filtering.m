function verify_goodness_filtering(Results, sim_indices, output_filename)
    %% FUNCTION DESCRIPTION
    % The function extracts specific simulations and runs the filter on the
    % given observations. The idea is to check whether the estimated
    % parameters are not too far from the true ones (which we know for
    % simulated data)
    % INPUTS:
    %   Results         : Structure containing simulation results (Pair_1.Mean, etc.)
    %   sim_indices     : (Optional) Vector of indices to check. Default: [1, 2]
    %   output_filename : (Optional) Name of the log file. Default: 'Results_Table.txt'
    %% ====================================================================

    %% 1. Default Arguments
    if nargin < 2 || isempty(sim_indices)
        sim_indices = [1, 2];
    end
    if nargin < 3 || isempty(output_filename)
        output_filename = 'Results_Table.txt';
    end

    %% 2. Open / create the file which will contain the results of the
    %           estimation in txt. format
    fid = fopen(output_filename, 'w');
    if fid == -1
        error('Cannot open the file %s', output_filename);
    end
    % Ensure file closes even if error occurs
    cleanupObj = onCleanup(@() fclose(fid));

    %% 3. Configuration
    % Rhos to try: 3rd dimension index in the "Results" tensor, 
    % value of rho, Label
    rho_scenarios = {
        1,   0.0,  'Exogenous';
        6,  -0.5,  'Rho = -0.5';
        11, -1.0,  'Rho = -1'
    };

    % Base parameters (True values excluding Rho)
    base_mean = [0.6; 3.0; 0.8; 0.4; 0.5; NaN; 0.5]; 
    names_mean = {'Mu_Low', 'Mu_High', 'Sigma', 'Alpha', 'Tau', 'Rho', 'Gamma'};

    base_vol  = [0.0; 0.04; 0.12; 0.4; 0.5; NaN];
    names_vol = {'Mu', 'Sigma_Low', 'Sigma_High', 'Alpha', 'Tau', 'Rho'};

    models_list = {'mean', 'volatility'};

    %% 4. Main Loop
    for m = 1:length(models_list)
        model_type = models_list{m};
        is_mean = strcmp(model_type, 'mean');
        
        % Model specific setup
        if is_mean
            k_lag = 1;
            run_type = 'mean';
            base_true_params = base_mean;
            param_names = names_mean;
            rho_pos = 6;    
            % When filtering, vector of params = [mu_low, mu_high, sigma,
            % alpha, tau, rho, gamma]
        else
            k_lag = 0;
            run_type = 'volatility'; 
            base_true_params = base_vol;
            param_names = names_vol;
            rho_pos = 6; 
        end
        
        % Header to print both in the txt file and the MATLAB window
        dual_print(fid, '\n%s\n', repmat('*', 1, 60));
        dual_print(fid, ' ANALYSIS: %s MODEL\n', upper(model_type));
        dual_print(fid, '%s\n', repmat('*', 1, 60));

        % Loop on the selected simulations
        for s_idx = 1:length(sim_indices)
            curr_sim = sim_indices(s_idx);
            
            % Loop on the different rhos
            for r = 1:size(rho_scenarios, 1)
                curr_rho_idx = rho_scenarios{r, 1};
                curr_rho_val = rho_scenarios{r, 2};
                curr_rho_lbl = rho_scenarios{r, 3};
                
                % Extract Data
                if is_mean
                    % Check bounds to avoid errors if Results is smaller
                    if curr_sim > size(Results.Pair_1.Mean, 2)
                        warning('Sim index %d out of bounds for Mean model. Skipping.', curr_sim);
                        continue;
                    end
                    y = Results.Pair_1.Mean(:, curr_sim, curr_rho_idx);
                else
                    if curr_sim > size(Results.Pair_1.Vol, 2)
                        warning('Sim index %d out of bounds for Vol model. Skipping.', curr_sim);
                        continue;
                    end
                    y = Results.Pair_1.Vol(:, curr_sim, curr_rho_idx);
                end
                
                % Processing Message (to know where we're at)
                fprintf('>> Processing: %s | Sim #%d | %s...\n', model_type, curr_sim, curr_rho_lbl);
                
                % Run Analysis
                % plotting set to false to speed up
                output = run_full_analysis(y, k_lag, run_type, true, false); 
                
                % Construct true parameters vector and then print it
                true_params = base_true_params;
                true_params(rho_pos) = curr_rho_val;
                title_str = sprintf('%s - Sim %d - %s', upper(model_type), curr_sim, curr_rho_lbl);
                print_verification_table(fid, title_str, output.params_estimated, true_params, output.se, param_names);
            end
        end
    end
    
    fprintf('\nAnalysis done. Results saved in %s\n', output_filename);
end

%% LOCAL HELPER FUNCTIONS

function dual_print(fid, varargin)
% Function that allows to print both in the txt. file and in the MATLAB
% window
    % Prints to command window
    fprintf(varargin{:});
    % Prints to file if valid
    if fid > 0
        fprintf(fid, varargin{:});
    end
end

function print_verification_table(fid, title_str, est_params, true_params, est_se, param_names)
    % Formats and prints the comparison table with t-stats and p-values
    
    dual_print(fid, '\n--- %s ---\n', title_str);
    dual_print(fid, '%-10s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n', ...
        'Param', 'True', 'Est', 'Diff', 'SE', 't-bias', 'p-val');
    dual_print(fid, '%s\n', repmat('-', 1, 85));
    
    for i = 1:length(param_names)
        p_name = param_names{i};
        p_true = true_params(i);
        p_est  = est_params(i);
        p_se   = est_se(i);
        
        p_diff = p_est - p_true;
        
        % t-stat value
        if p_se > 1e-10
            t_stat = (p_est - p_true) / p_se;
            % 3. P-value 
            p_val  = 2 * (1 - normcdf(abs(t_stat)));
        else
            t_stat = NaN;
            p_val  = NaN;
        end
        
        dual_print(fid, '%-10s | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f | %8.4f\n', ...
            p_name, p_true, p_est, p_diff, p_se, t_stat, p_val);
    end
    dual_print(fid, '%s\n', repmat('-', 1, 85));
end