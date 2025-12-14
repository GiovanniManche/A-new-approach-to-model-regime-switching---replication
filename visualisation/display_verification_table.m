function display_verification_table(title_str, output, true_vec, names)
%% FUNCTION DESCRIPTION
%  Display comparison tables between true value of parameters and estimated
% values. Useful to assess the potential bias with MC simulations
%% ========================================================================
    est_vec = output.params_estimated(:); % Colonne
    se_vec  = output.se(:);
    
    % Potential issues if different sizes
    len_true = length(true_vec);
    len_est  = length(est_vec);
    
    if len_est < len_true
        est_vec = [est_vec; nan(len_true - len_est, 1)];
        se_vec  = [se_vec; nan(len_true - len_est, 1)];
    elseif len_est > len_true
        est_vec = est_vec(1:len_true);
        se_vec = se_vec(1:len_true);
    end
    
    fprintf('\n=========================================================================\n');
    fprintf(' VERIFICATION: %s\n', title_str);
    fprintf('=========================================================================\n');
    fprintf('%-12s | %-10s | %-10s | %-10s | %-10s\n', 'Param', 'TRUE', 'ESTIMATED', 'BIAS', 'Z-Score');
    fprintf('-------------------------------------------------------------------------\n');

    for i = 1:len_true
        tv = true_vec(i);
        ev = est_vec(i);
        se = se_vec(i);
        
        bias = ev - tv;
        z_score = abs(bias) / se;
        
        quality = '';
        if ~isnan(z_score)
            if z_score < 1.96
                quality = '(OK)';
            elseif z_score < 3
                quality = '(!)';
            else
                quality = '(XX)';
            end
        end
        
        if isnan(ev)
            fprintf('%-12s | %10.4f | %10s | %10s | %10s\n', names{i}, tv, 'N/A', '-', '-');
        else
            fprintf('%-12s | %10.4f | %10.4f | %10.4f | %10.2f %s\n', ...
                names{i}, tv, ev, bias, z_score, quality);
        end
    end
    fprintf('-------------------------------------------------------------------------\n');
end