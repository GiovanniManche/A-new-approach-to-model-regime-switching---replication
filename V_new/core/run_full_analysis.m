function [out] = run_full_analysis(y, k, model_type, compare_exog, do_plot)
%% FUNCTION DESCRIPTION
% Runs estimation, testing, and filtering
% Modular wrapper for optimize_model and extract_results: depending on the
% model type and on the will or not to compare to exogenous regime
% switching estimation. Refer to RegimeSwitchingModel.m and 
% optimize_model.m for more details.
%% ========================================================================
    fprintf('\n========================================\n');
    fprintf('STARTING ANALYSIS: %s Model (k=%d)\n', upper(model_type), k);
    fprintf('========================================\n');

    %% Step A: Estimate Endogenous Model (Full)
    fprintf('1. Estimating Endogenous Model (Free Rho)...\n');
    res_endo = optimize_model(y, k, model_type); 
    
    best_model = res_endo;
    is_endogenous = true;

    %% Step B: Test against Exogenous (if compare_exog is TRUE)
    if compare_exog
        fprintf('2. Estimating Exogenous Model (Rho = 0)...\n');
        opts.fix_rho = 0;
        res_exo = optimize_model(y, k, model_type, opts);
        
        % Likelihood Ratio Test
        LR = 2 * (res_endo.loglik - res_exo.loglik);
        pval = 1 - chi2cdf(LR, 1);
        
        fprintf('   -> LR Stat: %.4f | P-value: %.4f\n', LR, pval);
        
        if pval > 0.10
            fprintf('   -> RESULT (Statistical): Exogenous model is sufficient (Rho not significant).\n');
            fprintf('   -> NOTE: Keeping Endogenous model to analyze time-varying probabilities.\n');
        else
            fprintf('   -> RESULT: Endogenous model is better (Rho is significant).\n');
        end
        
        out.comparison.lr_stat = LR;
        out.comparison.p_value = pval;
        out.comparison.res_exo = res_exo; % Fixed typo here (comparison)
    end

    %% Step C: Extract filtered probabilities (corollary 3.3)
    fprintf('3. Extracting filtered probabilities and transitions...\n');
    % Uses extract_results.m to get p00, p11, probabilities
    detailed_series = extract_results(best_model.params, y, k, model_type);
    
    %% Step D: Packaging Results
    out.y = y;
    out.params_estimated = best_model.params;
    out.loglik = best_model.loglik;
    out.se = best_model.se; % Standard errors
    out.filtered_probs = detailed_series.filtered_probs;
    out.p00 = detailed_series.p00; % Prob of staying in low regime
    out.p11 = detailed_series.p11; % Prob of staying in high regime
    out.is_endogenous = is_endogenous;
    
    %% Step E: Quick Visualization
    if do_plot
        plot_results(out);
    end
end
%% ========================================================================
function plot_results(out)
%% FUNCTION DESCRIPTION
% Plot the initial data (either simulated or real), the (filtered) 
% probability of being in the high regime, and the time-varying
% probabilities of staying in the low regime or high regime.
%% ========================================================================
    figure('Name', 'Analysis Results', 'Color', 'w');
    
    subplot(3,1,1);
    plot(out.y, 'k'); 
    title('Observed Data'); 
    grid on; xlim([1 length(out.y)]);
    
    subplot(3,1,2);
    % Shaded area for High Regime
    area(out.filtered_probs, 'FaceColor', [.8 .8 .8], 'EdgeColor', 'none'); hold on;
    plot(out.filtered_probs, 'r', 'LineWidth', 1.5);
    yline(0.5, 'k--');
    title('Filtered Probability (High Regime / Expansion)'); 
    ylim([0 1]); grid on; xlim([1 length(out.y)]);
    
    subplot(3,1,3);
    plot(out.p00, 'b'); hold on; plot(out.p11, 'r');
    legend('P(Stay Low)', 'P(Stay High)', 'Location', 'SouthWest');
    title('Time-Varying Transition Probabilities'); 
    ylim([0 1]); grid on; xlim([1 length(out.y)]);
end