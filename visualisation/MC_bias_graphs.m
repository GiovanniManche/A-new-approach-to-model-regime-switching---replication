function MC_bias_graphs(MC_Data, output_folder)
    %% FUNCTION DESCRIPTION
    % Generates and save charts for MC analysis
    %
    % INPUTS:
    %   MC_Data       : Structure containing MC results
    %   output_folder : (Optional) folder where the user wants to save the
    %   charts. Default: visualisation/graphs
    %% ====================================================================

    %% CONFIGURATION
    if nargin < 2 || isempty(output_folder)
        output_folder = fullfile('visualisation', 'graphs');
    end

    % Create the folder if doesn't exist
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
        fprintf('Folder created: %s\n', output_folder);
    end

    % Define graphic properties
    fig_props = {'Color', 'w', 'Position', [100, 100, 1000, 500]};
    rho_idx = 6; % Index of Rho in parameter vector

    %% 1. FIGURE: DENSITY OF RHO ESTIMATOR 
    models = {'Mean', 'Vol'};
    for m = 1:length(models)
        mod = models{m};
        
        figure('Name', [mod ' Model - Rho Density'], fig_props{:});
        hold on;
        
        colors = {'b', '#D95319', 'k'}; % Blue, Orange, Black
        scenarios = {'Rho_0', 'Rho_Mid', 'Rho_Endog'};
        legend_entries = {};
        
        for s = 1:length(scenarios)
            scen = scenarios{s};
            field = sprintf('%s_%s', mod, scen);
            
            if isfield(MC_Data, field)
                data = MC_Data.(field);
                est_rhos = data.estimates(rho_idx, :);
                true_rho = data.true_rho;
                
                % Kernel Density Estimation
                [f, xi] = ksdensity(est_rhos);
                
                % Plot
                plot(xi, f, 'LineWidth', 2, 'Color', colors{s});
                xline(true_rho, '--', 'Color', colors{s}, 'LineWidth', 1, 'HandleVisibility', 'off');
                
                legend_entries{end+1} = sprintf('True \\rho = %.1f', true_rho);
            end
        end
        
        title([mod ' Model: Finite Sample Distribution of \rho Estimator']);
        xlabel('Estimated Value of \rho');
        ylabel('Density');
        legend(legend_entries, 'Location', 'NorthWest');
        grid on;
        
        filename = ['Fig_Density_' mod '.png'];
        saveas(gcf, fullfile(output_folder, filename));
    end

    %% 2. FIGURE: BIAS AND RMSE 
    for m = 1:length(models)
        mod = models{m};
        
        true_rhos = [];
        biases = [];
        rmses = [];
        
        scenarios = {'Rho_0', 'Rho_Mid', 'Rho_Endog'};
        
        for s = 1:length(scenarios)
            field = sprintf('%s_%s', mod, scenarios{s});
            if isfield(MC_Data, field)
                true_rhos(end+1) = MC_Data.(field).true_rho;
                biases(end+1)    = MC_Data.(field).bias;
                rmses(end+1)     = MC_Data.(field).rmse;
            end
        end
        
        figure('Name', [mod ' Model - Bias/RMSE'], fig_props{:});
        
        subplot(1, 2, 1);
        plot(true_rhos, biases, '-o', 'LineWidth', 2, 'MarkerFaceColor', 'auto');
        yline(0, 'k--', 'LineWidth', 1);
        title('Bias of estimated \rho');
        xlabel('True \rho'); ylabel('Bias');
        grid on;
        
        subplot(1, 2, 2);
        plot(true_rhos, rmses, '-s', 'LineWidth', 2, 'MarkerFaceColor', 'auto', 'Color', 'r');
        title('Root Mean Square Error (RMSE)');
        xlabel('True \rho'); ylabel('RMSE');
        grid on;
        
        sgtitle([mod ' Model: Performance Statistics']);
        
        filename = ['Fig_Performance_' mod '.png'];
        saveas(gcf, fullfile(output_folder, filename));
    end

    %% 3. FIGURE: T-RATIO DISTRIBUTION UNDER H0 (Rho = 0)
    for m = 1:length(models)
        mod = models{m};
        field = sprintf('%s_Rho_0', mod); 
        
        if isfield(MC_Data, field)
            data = MC_Data.(field);
            
            est_rhos = data.estimates(rho_idx, :);
            se_rhos  = data.se(rho_idx, :);
            
            t_stats = est_rhos ./ se_rhos;
            
            figure('Name', [mod ' Model - T-Stat'], fig_props{:});
            hold on;
            
            histogram(t_stats, 25, 'Normalization', 'pdf', 'FaceColor', [.8 .8 .8], 'EdgeColor', 'none');
            
            [f, xi] = ksdensity(t_stats);
            plot(xi, f, 'b', 'LineWidth', 2);
            
            x_grid = linspace(min(t_stats), max(t_stats), 100);
            y_norm = normpdf(x_grid, 0, 1);
            plot(x_grid, y_norm, 'r--', 'LineWidth', 2);
            
            title(['Distribution of t-ratio for \rho=0 (' mod ' Model)']);
            xlabel('t-statistic'); ylabel('Density');
            legend('Simulation Histogram', 'Empirical Density', 'Standard Normal N(0,1)');
            grid on;
            
            filename = ['Fig_T_Stat_' mod '.png'];
            saveas(gcf, fullfile(output_folder, filename));
        end
    end
    
    fprintf('All figures saved in: %s\n', output_folder);
end