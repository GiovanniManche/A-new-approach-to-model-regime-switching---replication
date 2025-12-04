%% TEST compute_moments (r regimes, states in {0, 1, ..., r-1})
clear; clc;

fprintf('========================================\n');
fprintf('TEST: compute_moments (r regimes)\n');
fprintf('========================================\n\n');

%% Helper for printing configs
print_config = @(cfg) fprintf('Config: [ %s]\n', sprintf('%d ', cfg));

%% Test 1: Volatility model (no AR, gamma=0, r=2, k=0)
fprintf('Test 1: Volatility model (gamma=0)\n');
fprintf('-----------------------------------\n');

r = 2; 
k = 0;

mu_vec    = [0; 0];
sigma_vec = [0.04; 0.12];
gamma_vec = [];   % no AR

params = create_params(0.8, 0.0, -0.5, mu_vec, gamma_vec, sigma_vec, k);

y = randn(10,1);

% Config: only s_t since k = 0 → length(config) = 1
% s_t = 0 (low regime)
config = [0];

[m_t, sigma_t] = compute_moments(y, config, 5, params);

print_config(config);
fprintf('m_t      = %.4f (expected: %.4f)\n', m_t, params.mu(1));
fprintf('sigma_t  = %.4f (expected: %.4f)\n', sigma_t, params.sigma(1));

assert(m_t == params.mu(1));
assert(sigma_t == params.sigma(1));

fprintf('[OK] Test 1 passed\n\n');

%% Test 2: Volatility model, regime 1 (high)
fprintf('Test 2: Volatility model, regime 1 (high)\n');
fprintf('------------------------------------------\n');

config = [1];   % s_t = 1 (high regime)

[m_t, sigma_t] = compute_moments(y, config, 5, params);

print_config(config);
fprintf('m_t      = %.4f (expected: %.4f)\n', m_t, params.mu(2));
fprintf('sigma_t  = %.4f (expected: %.4f)\n', sigma_t, params.sigma(2));

assert(m_t == params.mu(2));
assert(sigma_t == params.sigma(2));

fprintf('[OK] Test 2 passed\n\n');

%% Test 3: Mean model with AR(1), r=2, k=1
fprintf('Test 3: Mean model with AR(1)\n');
fprintf('------------------------------\n');

r = 2; 
k = 1;

mu_vec    = [0.6; 3.0];
sigma_vec = [0.8; 0.8];
gamma_vec = 0.5;

params = create_params(0.8, 0.0, -0.5, mu_vec, gamma_vec, sigma_vec, k);

y = [1.0; 1.5; 2.0; 2.5; 3.0];

% Config: [s_t, s_{t-1}] = [0, 1] (low, high)
config = [0, 1];
t = 3;

[m_t, sigma_t] = compute_moments(y, config, t, params);

% Manual:
% s_t = 0 → mu_t    = mu_vec(1) = 0.6
% s_{t-1} = 1 → mu_{t-1} = mu_vec(2) = 3.0
% m_t = 0.6 + 0.5 * (y(2) - 3.0) = 0.6 + 0.5*(1.5 - 3.0)
expected_m_t = 0.6 + 0.5 * (y(t-1) - 3.0);

print_config(config);
fprintf('t        = %d\n', t);
fprintf('y(t-1)   = %.2f\n', y(t-1));
fprintf('m_t      = %.4f (expected: %.4f)\n', m_t, expected_m_t);
fprintf('sigma_t  = %.4f (expected: %.4f)\n', sigma_t, params.sigma(1));

assert(abs(m_t - expected_m_t) < 1e-10);
assert(sigma_t == params.sigma(1));

fprintf('[OK] Test 3 passed\n\n');

%% Test 4: AR(1) both regimes low (0, 0)
fprintf('Test 4: AR(1), both regimes low\n');
fprintf('--------------------------------\n');

config = [0, 0];  % Both in low regime
t = 4;

[m_t, sigma_t] = compute_moments(y, config, t, params);

% mu_t    = mu(1) = 0.6
% mu_{t-1}= mu(1) = 0.6
expected_m_t = 0.6 + 0.5 * (y(t-1) - 0.6);

print_config(config);
fprintf('t        = %d\n', t);
fprintf('y(t-1)   = %.2f\n', y(t-1));
fprintf('m_t      = %.4f (expected: %.4f)\n', m_t, expected_m_t);

assert(abs(m_t - expected_m_t) < 1e-10);

fprintf('[OK] Test 4 passed\n\n');

%% Test 5: AR(2) model, r=2, k=2
fprintf('Test 5: AR(2) model (k=2)\n');
fprintf('-------------------------\n');

k = 2;
gamma_vec = [0.5; 0.3];
params = create_params(0.8, 0.0, -0.5, mu_vec, gamma_vec, sigma_vec, k);

% Config: [s_t, s_{t-1}, s_{t-2}] = [1, 0, 1] (high, low, high)
config = [1, 0, 1];
t = 5;

[m_t, sigma_t] = compute_moments(y, config, t, params);

% Manual:
% s_t = 1 → mu_t    = mu_vec(2) = 3.0
% s_{t-1} = 0 → mu_{t-1} = mu_vec(1) = 0.6
% s_{t-2} = 1 → mu_{t-2} = mu_vec(2) = 3.0
% m_t = 3.0 + 0.5*(y(4)-0.6) + 0.3*(y(3)-3.0)
expected_m_t = 3.0 + 0.5*(y(4) - 0.6) + 0.3*(y(3) - 3.0);

print_config(config);
fprintf('t        = %d\n', t);
fprintf('m_t      = %.4f (expected: %.4f)\n', m_t, expected_m_t);
fprintf('sigma_t  = %.4f (expected: %.4f)\n', sigma_t, params.sigma(2));

assert(abs(m_t - expected_m_t) < 1e-10);
assert(sigma_t == params.sigma(2));

fprintf('[OK] Test 5 passed\n\n');

%% Test 6: Edge case t=1 (no lags)
fprintf('Test 6: Edge case t=1 (no lags)\n');
fprintf('--------------------------------\n');

k = 1;
params = create_params(0.8, 0.0, -0.5, mu_vec, 0.5, sigma_vec, k);

config = [0, 0];  % length k+1 = 2
t = 1;

[m_t, sigma_t] = compute_moments(y, config, t, params);

% No lags available, so m_t = mu(s_t) = mu(1)
expected_m_t = mu_vec(1);

print_config(config);
fprintf('t        = %d\n', t);
fprintf('m_t      = %.4f (expected: %.4f)\n', m_t, expected_m_t);

assert(m_t == expected_m_t);

fprintf('[OK] Test 6 passed\n\n');

%% Test 7: ERROR - invalid config length
fprintf('Test 7: ERROR expected (wrong config length)\n');
fprintf('---------------------------------------------\n');

try
    config_bad = [0];  % k=1 => need length 2
    [m_t, sigma_t] = compute_moments(y, config_bad, 3, params);
    fprintf('[KO] PROBLEM: should have raised an error!\n');
catch ME
    fprintf('[OK] Error correctly detected: %s\n\n', ME.message);
end

%% Test 8: ERROR - invalid state values (out of range)
fprintf('Test 8: ERROR expected (invalid state values)\n');
fprintf('----------------------------------------------\n');

try
    config_bad = [0, 2];  % states must be in {0,1} for r=2
    [m_t, sigma_t] = compute_moments(y, config_bad, 3, params);
    fprintf('[KO] PROBLEM: should have raised an error!\n');
catch ME
    fprintf('[OK] Error correctly detected: %s\n\n', ME.message);
end

%% Test 9: ERROR - negative state values
fprintf('Test 9: ERROR expected (negative state values)\n');
fprintf('-----------------------------------------------\n');

try
    config_bad = [-1, 0];  % negative state
    [m_t, sigma_t] = compute_moments(y, config_bad, 3, params);
    fprintf('[KO] PROBLEM: should have raised an error!\n');
catch ME
    fprintf('[OK] Error correctly detected: %s\n\n', ME.message);
end

%% Test 10: Verify all configs for r=2, k=1
fprintf('Test 10: All configurations for r=2, k=1\n');
fprintf('-----------------------------------------\n');

k = 1;
params = create_params(0.8, 0.0, -0.5, mu_vec, 0.5, sigma_vec, k);

configs = generate_all_configs(k, r);

fprintf('Testing all %d configurations:\n', size(configs, 1));

for i = 1:size(configs, 1)
    config = configs(i, :);
    
    try
        [m_t, sigma_t] = compute_moments(y, config, 3, params);
        fprintf('  Config [%d %d]: m_t=%.4f, sigma_t=%.4f [OK]\n', ...
            config(1), config(2), m_t, sigma_t);
    catch ME
        fprintf('  Config [%d %d]: [KO] %s\n', ...
            config(1), config(2), ME.message);
    end
end

fprintf('\n[OK] Test 10 passed\n\n');

%% Summary
fprintf('========================================\n');
fprintf('ALL TESTS PASSED (compute_moments)\n');
fprintf('========================================\n');