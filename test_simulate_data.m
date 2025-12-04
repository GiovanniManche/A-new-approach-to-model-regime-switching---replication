function test_simulate_data()
% TEST_SIMULATE_DATA
%   Quick diagnostics for simulate_data.m:
%   - checks regime-dependent means/volatilities
%   - checks endogeneity: corr(u_{t-1}, v_t) ≈ rho
%
%   Assumes you have:
%       [y, s, w, u] = simulate_data(n, params)

    clc; close all;
    fprintf('=====================================\n');
    fprintf('   TEST: simulate_data\n');
    fprintf('=====================================\n\n');

    rng(0);  % for reproducibility

    %% 1. Set parameters (you can change these)
    params.alpha      = 1;     % latent AR(1) coef
    params.tau        = 0.95;     % threshold
    params.rho        = -0.2;    % endogeneity

    % Regime-dependent means
    params.mu_low     = 0.6;
    params.mu_high    = 3.0;

    % AR order and coeffs (AR(1) around the regime-dependent mean)
    params.k          = 1;
    params.gamma      = 0.5;     % scalar -> AR(1)

    % Regime-dependent volatilities
    % To mimic CCP mean model: same sigma in both regimes
    params.sigma_low  = 0.8;
    params.sigma_high = 0.8;

    n = 3000;   % sample size

    %% 2. Simulate data
    fprintf('Simulating %d observations...\n', n);
    [y, s, w, u] = simulate_data(n, params);

    % Reconstruct v_t = w_t - alpha * w_{t-1}
    v = nan(n,1);
    v(2:end) = w(2:end) - params.alpha * w(1:end-1);

    %% 3. Plots: y, w, s
    figure;
    subplot(3,1,1);
    plot(y, 'k-');
    title('Simulated y_t');
    xlabel('t');

    subplot(3,1,2);
    plot(w, 'b-');
    hold on;
    yline(params.tau, 'r--', 'Threshold \tau');
    title('Latent factor w_t');
    xlabel('t');

    subplot(3,1,3);
    stairs(s, 'r-');
    ylim([-0.2 1.2]);
    title('Regime s_t');
    xlabel('t');

    %% 4. Regime-dependent moments for y
    idx0 = (s == 0);
    idx1 = (s == 1);

    mean_low_emp  = mean(y(idx0));
    mean_high_emp = mean(y(idx1));
    std_low_emp   = std(y(idx0));
    std_high_emp  = std(y(idx1));

    fprintf('\n--- Regime-dependent moments ---\n');
    fprintf('mu_low   (true) = %6.3f   |  mean(y | s=0) = %6.3f\n', ...
            params.mu_low,  mean_low_emp);
    fprintf('mu_high  (true) = %6.3f   |  mean(y | s=1) = %6.3f\n', ...
            params.mu_high, mean_high_emp);
    fprintf('sigma_low (true) = %6.3f  |  std(y  | s=0) = %6.3f\n', ...
            params.sigma_low,  std_low_emp);
    fprintf('sigma_high(true) = %6.3f  |  std(y  | s=1) = %6.3f\n', ...
            params.sigma_high, std_high_emp);

    %% 5. Endogeneity check: corr(u_{t-1}, v_t)
    valid = 2:n;  % v(2..n) exists and corresponds to u(1..n-1)
    emp_rho = corr(u(valid-1), v(valid), 'rows', 'complete');

    fprintf('\n--- Endogeneity check ---\n');
    fprintf('rho (true)       = %6.3f\n', params.rho);
    fprintf('corr(u_{t-1},v_t)= %6.3f\n', emp_rho);

    %% 6. Simple pass/fail messages
    tol_mean  = 0.2;
    tol_sigma = 0.2;
    tol_rho   = 0.1;

    if abs(mean_low_emp - params.mu_low) < tol_mean && ...
       abs(mean_high_emp - params.mu_high) < tol_mean
        fprintf('\n[OK] Regime means are close to target values.\n');
    else
        fprintf('\n[WARN] Regime means are far from targets. Check simulate_data.\n');
    end

    if abs(std_low_emp - params.sigma_low) < tol_sigma && ...
       abs(std_high_emp - params.sigma_high) < tol_sigma
        fprintf('[OK] Regime std devs are close to target values.\n');
    else
        fprintf('[WARN] Regime std devs are far from targets.\n');
    end

    if abs(emp_rho - params.rho) < tol_rho
        fprintf('[OK] Endogeneity corr(u_{t-1}, v_t) matches rho.\n');
    else
        fprintf('[WARN] corr(u_{t-1}, v_t) is far from rho. Check v_t construction.\n');
    end

    fprintf('\nTest finished.\n');

end
