function [tables] = display_estimation_results(output, options)
    %% FUNCTION DESCRIPTION
    % Display in the matlab window and save in txt file estimation results.
    % 
    % INPUTS :
    %   output      : Structure from run_full_analysis containing:
    %                   .params_estimated : Estimated parameters
    %                   .se               : Standard errors
    %                   .loglik           : Log-likelihood
    %                   .comparison       : (Optional) LR test results
    %
    %   options    : (Optional) Structure with fields:
    %                   .save_to_file  : Filename to save results (ex: 'results.txt')
    %                                    (If empty, no file saved. Default: '')
    %                   .k             : Number of AR lags (for parameter naming)
    %                                    (Default: auto-detect from params length)
    %                   .model_name    : Name to display in header
    %                                    (Default: 'REGIME SWITCHING MODEL')
    %                   .display       : Display in console (true/false)
    %                                    (Default: true)
    %                   .precision     : Number of decimals for estimates
    %                                    (Default: 4)
    %
    % OUTPUTS :
    %   tables     :    Structure containing:
    %                       .endogenous  : Table for endogenous model
    %                       .exogenous   : Table for exogenous model (if available)
    %                       .comparison  : LR test results table
    %                       .text        : Full text output (same as file)
    %% ====================================================================
    
    %% Parse inputs + robustness
    if nargin < 2
        options = struct();
    end
    
    % Default options
    if ~isfield(options, 'save_to_file'), options.save_to_file = ''; end
    if ~isfield(options, 'display'), options.display = true; end
    if ~isfield(options, 'precision'), options.precision = 4; end
    if ~isfield(options, 'model_name'), options.model_name = 'REGIME SWITCHING MODEL'; end
    if ~isfield(options, 'model_type'), options.model_type = 'mean'; end
    
    % Auto-detect k if not provided
    if ~isfield(options, 'k')
        % Order: [Mu_L, Mu_H, Sigma, Alpha, Tau, Rho, Gamma_1...Gamma_k]
        n_params = length(output.params_estimated);
        options.k = n_params - 6;  % Subtract fixed params
    end
    
    k = options.k;
    
    %% Extract parameters
    est = output.params_estimated;
    se = output.se;
    loglik = output.loglik;
    
    % Calculate t-stats
    t_stats = est ./ se;
    
    %% Generate parameter names dynamically
    % i.e. if mean model, [mu_low, mu_high, sigma,...]
    % but if vol model, [mu, sigma_low, sigma_high,...]
    if strcmpi(options.model_type, 'volatility')
        % VOLATILITY MODEL 
        % Order: [Mu, Sigma_L, Sigma_H, Alpha, Tau, Rho]
        param_names = { ...
            'Mu (Constant)', ...
            'Sigma (Low/Normal)', ...
            'Sigma (High/Crisis)', ...
            'Alpha (Latent)', ...
            'Tau (Threshold)', ...
            'Rho (Endogeneity)'
        };
    else
        % MEAN MODEL 
        % Order: [Mu_L, Mu_H, Sigma, Alpha, Tau, Rho]
        param_names = { ...
            'Mu (Recession)', ...
            'Mu (Expansion)', ...
            'Sigma (Constant)', ...
            'Alpha (Latent)', ...
            'Tau (Threshold)', ...
            'Rho (Endogeneity)'
        };
    end
    for j = 1:k
        param_names{end+1} = sprintf('Gamma (AR%d)', j);
    end
    
    %% Build endogenous model table
    table_endo = build_results_table(est, se, t_stats, param_names, ...
                                     loglik, 'ENDOGENOUS MODEL (Chang et al.)', ...
                                     options.precision);
    
    %% Build exogenous model table (if available)
    table_exo = '';
    if isfield(output, 'comparison') && isfield(output.comparison, 'res_exo')
        res_exo = output.comparison.res_exo;
        est_exo = res_exo.params;
        se_exo = res_exo.se;
        t_exo = est_exo ./ se_exo;
      
        param_names_exo = param_names;
        param_names_exo{6} = 'Rho (FIXED=0)';
        
        table_exo = build_results_table(est_exo, se_exo, t_exo, param_names_exo, ...
                                         res_exo.loglik, 'EXOGENOUS MODEL (Hamilton)', ...
                                         options.precision);
    end
    
    %% Build LR test table (if available)
    table_lr = '';
    if isfield(output, 'comparison')
        table_lr = build_lr_test_table(output.comparison, options.precision);
    end
    
    %% Assemble full output
    separator = repmat('=', 1, 70);
    header = sprintf('\n%s\n %s\n%s\n', separator, options.model_name, separator);
    
    full_text = [header, table_endo];
    
    if ~isempty(table_exo)
        full_text = [full_text, sprintf('\n\n'), table_exo];
    end
    
    if ~isempty(table_lr)
        full_text = [full_text, sprintf('\n\n'), table_lr];
    end
    
    full_text = [full_text, sprintf('\n%s\n', separator)];
    
    %% Display to MATLAB
    if options.display
        fprintf('%s', full_text);
    end
    
    %% Save to file
    if ~isempty(options.save_to_file)
        try
            fid = fopen(options.save_to_file, 'w');
            fprintf(fid, '%s', full_text);
            fclose(fid);
            fprintf('\n✓ Results saved to: %s\n', options.save_to_file);
        catch ME
            warning('Failed to save file: %s', ME.message);
        end
    end
    
    %% Package output
    tables.endogenous = table_endo;
    tables.exogenous = table_exo;
    tables.comparison = table_lr;
    tables.text = full_text;
    
    tables.table_endo = create_matlab_table(est, se, t_stats, param_names);
    if ~isempty(table_exo)
        tables.table_exo = create_matlab_table(est_exo, se_exo, t_exo, param_names_exo);
    end
    
    end
    
    %% ========================================================================
    %% HELPER FUNCTIONS
    %% ========================================================================
    
    function table_str = build_results_table(est, se, t_stats, param_names, loglik, title_str, precision)
        % Build formatted results table
        
        n = length(est);
        
        % Header
        sep = repmat('-', 1, 70);
        table_str = sprintf('%s\n', sep);
        table_str = [table_str, sprintf(' %s\n', title_str)];
        table_str = [table_str, sprintf('%s\n', sep)];
        table_str = [table_str, sprintf('%-20s | %-12s | %-12s | %-10s\n', ...
                                        'Parameter', 'Estimate', 'Std.Err', 't-Stat')];
        table_str = [table_str, sprintf('%s\n', sep)];
        
        % Format string for estimates
        fmt_est = sprintf('%%12.%df', precision);
        fmt_se = sprintf('%%12.%df', precision);
        
        % Rows
        for i = 1:n
            val = est(i);
            std_err = se(i);
            ts = t_stats(i);
            
            % Handle fixed parameters 
            if std_err < 1e-6
                row = sprintf('%-20s | %s | %12s | %10s\n', ...
                             param_names{i}, ...
                             sprintf(fmt_est, val), ...
                             '(Fixed)', ...
                             '-');
            else
                % Significance stars
                sig_star = get_significance_stars(ts);
                
                row = sprintf('%-20s | %s | %s | %10.2f %s\n', ...
                             param_names{i}, ...
                             sprintf(fmt_est, val), ...
                             sprintf(fmt_se, std_err), ...
                             ts, sig_star);
            end
            
            table_str = [table_str, row];
        end
        
        % Footer
        table_str = [table_str, sprintf('%s\n', sep)];
        table_str = [table_str, sprintf('Log-Likelihood : %.4f\n', loglik)];
        table_str = [table_str, sprintf('%s\n', sep)];
        
        % Legend
        table_str = [table_str, sprintf('Significance: *** p<0.01, ** p<0.05, * p<0.10\n')];
    end
    
    function table_str = build_lr_test_table(comparison, precision)
        % Build LR test results table
        
        sep = repmat('-', 1, 70);
        
        table_str = sprintf('%s\n', sep);
        table_str = [table_str, sprintf(' LIKELIHOOD RATIO TEST (Endogeneity Check)\n')];
        table_str = [table_str, sprintf('%s\n', sep)];
        table_str = [table_str, sprintf('H0: Rho = 0 (Exogenous model is sufficient)\n\n')];
        
        lr_stat = comparison.lr_stat;
        p_value = comparison.p_value;
        
        table_str = [table_str, sprintf('LR Statistic : %.*f\n', precision, lr_stat)];
        table_str = [table_str, sprintf('P-Value      : %.*f\n', precision, p_value)];
        table_str = [table_str, sprintf('Df           : 1\n\n')];
        
        % Decision
        if p_value < 0.01
            decision = 'REJECT H0 at 1%% level. Strong evidence for endogeneity. ***';
        elseif p_value < 0.05
            decision = 'REJECT H0 at 5%% level. Endogeneity is significant. **';
        elseif p_value < 0.10
            decision = 'REJECT H0 at 10%% level. Weak evidence for endogeneity. *';
        else
            decision = 'FAIL TO REJECT H0. No strong evidence for endogeneity.';
        end
        
        table_str = [table_str, sprintf('Decision: %s\n', decision)];
        table_str = [table_str, sprintf('%s\n', sep)];
    end
    
    function stars = get_significance_stars(t_stat)
        % Return significance stars based on t-statistic
        
        abs_t = abs(t_stat);
        
        if abs_t > 2.576
            stars = '***';  % 1%
        elseif abs_t > 1.96
            stars = '**';   % 5%
        elseif abs_t > 1.645
            stars = '*';    % 10%
        else
            stars = '';
        end
    end
    
    function tbl = create_matlab_table(est, se, t_stats, param_names)
        % Create table object for easy export
        
        n = length(est);
        
        % Prepare data
        Parameter = param_names(:);
        Estimate = est(:);
        StdErr = se(:);
        tStat = t_stats(:);
        
        % Significance
        Significance = cell(n, 1);
        for i = 1:n
            Significance{i} = get_significance_stars(t_stats(i));
        end
        
        % Create table
        tbl = table(Parameter, Estimate, StdErr, tStat, Significance);

    end

