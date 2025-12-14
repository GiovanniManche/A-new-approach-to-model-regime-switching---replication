function [out] = run_full_analysis(y, k, model_type, compare_exog, do_plot, varargin)
    %% FUNCTION DESCRIPTION
    % Modular wrapper for optimize_model and extract_results: depending on the
    % model type and on the will or not to compare to exogenous regime
    % switching estimation. Refer to RegimeSwitchingModel.m and
    % optimize_model.m for more details.
    %
    % INPUTS:
    %   y               : (T x 1) time series
    %   k               : order of the AR on the time series (0 if volatility model)
    %   model_type      : either 'mean' or 'volatility' (please be sure of
    %   matching the case)
    %   compare_exog    : boolean, if true exogenous filter is executed and
    %                     likelihood ratio test is done.
    %   do_plot         : boolean, if true charts containing observed data, filtered
    %                     probabilities, latent factor are given
    %   varagin         : (Optional)
    %                        'dates'        : Vector of dates (datetime or numeric)
    %                                 If not provided, uses indices 1:T
    %                        'save_figs'    : Save figures to disk 
    %                                         (default: false)
    %                        'fig_path'     : Directory for saved figures 
    %                                         (default: 'visualisation/graphs')
    %                        'fig_name'     : Prefix for figure names 
    %                                         (default: 'model_results')
    %                        'fig_format'   : Format for saving ('png', 'pdf', 'eps', 'fig')
    %                                         (default: {'png', 'fig'})
    %
    % OUTPUT:
    %   out: structure containing useful elements for analysis / plotting:
    %       y                   : time series
    %       dates               : dates associated with the time series
    %       params_estimated    : values of endogenous model's parameters
    %       loglik              : log-likelihood associated to the endogenous model
    %       se                  : standard errors of coefficients
    %       filtered_probs      : filtered probabilities of being in the low
    %                             regime
    %       p00                 : time-varying probabilities of staying in the low regime
    %       p11                 : time-varying probabilities of staying in the high regime
    %       latent_w            : infered latent factor
    %% ====================================================================

    %% 1. Parse optional inputs
    p = inputParser;
    addParameter(p, 'dates', [], @(x) isnumeric(x) || isdatetime(x));
    addParameter(p, 'save_figs', false, @islogical);
    addParameter(p, 'fig_path', 'visualisation/graphs', @ischar);
    addParameter(p, 'fig_name', 'model_results', @ischar);
    addParameter(p, 'fig_format', {'png', 'fig'}, @(x) ischar(x) || iscell(x));
    parse(p, varargin{:});
    
    dates = p.Results.dates;
    save_figs = p.Results.save_figs;
    fig_path = p.Results.fig_path;
    fig_name = p.Results.fig_name;
    fig_format = p.Results.fig_format;
    
    % Ensure fig_format is cell
    if ischar(fig_format)
        fig_format = {fig_format};
    end
    
    % Create dates vector if not provided
    T = length(y);
    if isempty(dates)
        dates = 1:T;
        has_dates = false;
    else
        has_dates = true;
        if length(dates) ~= T
            error('Length of dates must match length of y');
        end
    end

    fprintf('\n========================================\n');
    fprintf('STARTING ANALYSIS: %s Model (k=%d)\n', upper(model_type), k);
    fprintf('========================================\n');
    
    %% 2. Estimate Endogenous Model
    % Log for user
    fprintf('1. Estimating Endogenous Model (Free Rho)...\n');
    res_endo = optimize_model(y, k, model_type);
    
    best_model = res_endo;
    is_endogenous = true;

    %% 2. Test against Exogenous (if compare_exog is TRUE)
    if compare_exog
        % Log for user
        fprintf('2. Estimating Exogenous Model (Rho = 0)...\n');
        % Cf optimize_model documentation, but this fixing forces the model
        % to have rho exogenous.
        opts.fix_rho = 0;
        res_exo = optimize_model(y, k, model_type, opts);
        
        % LR Test (we suppose asymptotic validity)
        LR = 2 * (res_endo.loglik - res_exo.loglik);
        pval = 1 - chi2cdf(LR, 1);
        
        fprintf('LR Stat: %.4f | P-value: %.4f\n', LR, pval);
        fprintf('H0: rho = 0');
        if pval > 0.10
            fprintf('RESULT: Exogenous model is sufficient (Rho not significant).\n');
            fprintf('NOTE: Keeping Endogenous model to analyze time-varying probabilities.\n');
        else
            fprintf('RESULT: Endogenous model is better (Rho is significant).\n');
        end
        
        out.comparison.lr_stat = LR;
        out.comparison.p_value = pval;
        out.comparison.res_exo = res_exo;
    end

    %% 3. Extract filtered probabilities (corollary 3.3)
    fprintf('3. Extracting infered elements (probabilities, latent factor...)\n');
    detailed_series = extract_results(best_model.params, y, k, model_type);
    
    %% 4. Packaging Results
    out.y = y;
    out.dates = dates;
    out.params_estimated = best_model.params;
    out.loglik = best_model.loglik;
    out.se = best_model.se;
    out.filtered_probs = detailed_series.filtered_probs;
    out.p00 = detailed_series.p00;
    out.p11 = detailed_series.p11;
    out.latent_w = detailed_series.latent_w;
    out.is_endogenous = is_endogenous;
    
    % Compute p10 = 1 - p11 (prob of switching from high to low)
    out.p10 = 1 - detailed_series.p11;
    
    %% Step E: Quick Visualization
    if do_plot
        figs = plot_results(out, has_dates);
        
        % Save figures if requested
        if save_figs
            save_figures(figs, fig_path, fig_name, fig_format);
        end
    end
end

%% ========================================================================
function figs = plot_results(out, has_dates)
%% FUNCTION DESCRIPTION
% Plot the initial data (either simulated or real), the (filtered)
% probability of being in the recession regime, and the time-varying
% transition probabilities separately for p00 and p10.
%% ========================================================================

    % Extract dates for plotting
    if has_dates
        x_axis = out.dates;
        x_label = 'Date';
    else
        x_axis = 1:length(out.y);
        x_label = 'Time';
    end
    
    % Adjust for k lags (filtered probs start at k+1)
    k = length(out.y) - length(out.filtered_probs);
    x_axis_filtered = x_axis(k+1:end);
    
    %% Figure 1: Data and Recession Probability
    figs(1) = figure('Name', 'Data and Recession Probability', ...
                     'Color', 'w', 'Position', [100, 100, 1000, 600]);
    
    % Subplot 1: Observed Data
    subplot(2,1,1);
    plot(x_axis, out.y, 'k', 'LineWidth', 1.5);
    title('Observed Data', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Value');
    grid on;
    if has_dates
        datetick('x', 'keeplimits');
    end
    
    % Subplot 2: Recession Probability (P(S_t = 0))
    subplot(2,1,2);
    prob_recession = 1 - out.filtered_probs;  % P(S_t = 0) = 1 - P(S_t = 1)
    
    % Shaded area for Recession
    area(x_axis_filtered, prob_recession, ...
         'FaceColor', [0.7 0.7 0.9], 'EdgeColor', 'none'); hold on;
    plot(x_axis_filtered, prob_recession, 'b', 'LineWidth', 2);
    
    title('Filtered Probability of Low Regime', ...
          'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Probability');
    xlabel(x_label);
    ylim([0 1]);
    grid on;
    if has_dates
        datetick('x', 'keeplimits');
    end
    
    %% Figure 2: Transition Probabilities
    figs(2) = figure('Name', 'Time Varying Transition Probabilities', ...
                     'Color', 'w', 'Position', [150, 150, 1000, 600]);
    
    % Subplot 1:Stay in Recession
    subplot(2,1,1);
    plot(x_axis_filtered, out.p00, 'b', 'LineWidth', 2);
    title('Probability of Staying in Low State', ...
          'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Probability');
    ylim([0 1]);
    grid on;
    if has_dates
        datetick('x', 'keeplimits');
    end
    
    % Subplot 2: Switch to Recession
    subplot(2,1,2);
    plot(x_axis_filtered, out.p10, 'r', 'LineWidth', 2);
    title('Probability of Switching to Low State', ...
          'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Probability');
    xlabel(x_label);
    ylim([0 1]);
    grid on;
    if has_dates
        datetick('x', 'keeplimits');
    end
    
    
    %% Figure 3: Latent Factor (if available)
    if isfield(out, 'latent_w')
        figs(3) = figure('Name', 'Latent Factor', ...
                         'Color', 'w', 'Position', [200, 200, 1000, 400]);
        
        plot(x_axis_filtered, out.latent_w, 'b-', 'LineWidth', 1.5);
        hold on;
        
        % Add threshold
        tau = out.params_estimated(5);
        yline(tau, 'r--', 'LineWidth', 2, ...
              'Label', sprintf('\\tau = %.2f', tau), ...
              'LabelHorizontalAlignment', 'left');
        
        title('Latent Factor', 'FontSize', 12, 'FontWeight', 'bold');
        ylabel('w_t');
        xlabel(x_label);
        grid on;
        if has_dates
            datetick('x', 'keeplimits');
        end
        
        % Add legend
        legend('w_t', '\tau (threshold)', 'Location', 'best');
    end
end

%% ========================================================================
function save_figures(figs, fig_path, fig_name, fig_formats)
%% FUNCTION DESCRIPTION
% Save figures to specified directory in multiple formats
%% ========================================================================

    % Create directory if it doesn't exist
    if ~exist(fig_path, 'dir')
        mkdir(fig_path);
        fprintf('Created directory: %s\n', fig_path);
    end
    
    % Figure names
    fig_names = {
        'data_and_recession_prob'
        'transition_probabilities'
        'latent_factor'
    };
    
    fprintf('\nSaving figures to: %s\n', fig_path);
    
    for i = 1:length(figs)
        if isvalid(figs(i))  % Check if figure is still valid
            
            % Base filename
            base_name = fullfile(fig_path, sprintf('%s_%s', fig_name, fig_names{i}));
            
            % Save in each requested format
            for j = 1:length(fig_formats)
                fmt = fig_formats{j};
                filename = sprintf('%s.%s', base_name, fmt);
                
                try
                    switch lower(fmt)
                        case 'fig'
                            savefig(figs(i), filename);
                        case 'png'
                            print(figs(i), filename, '-dpng', '-r300');
                        case 'pdf'
                            print(figs(i), filename, '-dpdf', '-r300');
                        case 'eps'
                            print(figs(i), filename, '-depsc', '-r300');
                        case 'svg'
                            print(figs(i), filename, '-dsvg');
                        otherwise
                            warning('Unknown format: %s', fmt);
                            continue;
                    end
                    fprintf('Saved: %s\n', filename);
                catch ME
                    warning('Failed to save %s: %s', filename, ME.message);
                end
            end
        end
    end
    
    fprintf('Figure export complete.\n\n');
end