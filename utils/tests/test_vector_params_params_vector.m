%% TEST of params_to_vector() and vector_to_params()
clear; clc;

fprintf('========================================\n');
fprintf('TEST: Parameter Vector Conversion (r regimes)\n');
fprintf('========================================\n\n');

%% Test 1: Round-trip conversion (r=2, k=1)
fprintf('Test 1: Round-trip conversion (r=2, k=1)\n');
fprintf('----------------------------------------\n');

r = 2;
k = 1;

mu_vec    = [0.6; 3.0];
sigma_vec = [0.04; 0.12];
gamma_vec = 0.5;         % AR(1)

% Create parameter structure
params_original = create_params(0.8, 0.0, -0.5, mu_vec, gamma_vec, sigma_vec, k);

fprintf('Original parameters:\n');
disp(params_original);

% Structure → vector
theta_vec = params_to_vector(params_original);

expected_len = 3 + r + k + r;  % alpha,tau,rho,mu(1:r),gamma(1:k),sigma(1:r)

fprintf('Parameter vector:\n');
fprintf('  Length: %d (expected: %d for r=%d, k=%d)\n', ...
        length(theta_vec), expected_len, r, k);
fprintf('  Values: ');
fprintf('%.4f ', theta_vec);
fprintf('\n\n');

assert(length(theta_vec) == expected_len, 'Incorrect vector length.');

% Vector → structure
params_recovered = vector_to_params(theta_vec, k, r);

fprintf('Recovered parameters:\n');
disp(params_recovered);

% Equality checks
assert(params_recovered.alpha == params_original.alpha);
assert(params_recovered.tau   == params_original.tau);
assert(params_recovered.rho   == params_original.rho);
assert(all(params_recovered.mu    == params_original.mu));
assert(all(params_recovered.sigma == params_original.sigma));
assert(all(params_recovered.gamma == params_original.gamma));
assert(params_recovered.k    == params_original.k);
assert(params_recovered.r    == params_original.r);

fprintf('[OK] Perfect conversion (r=2, k=1)\n\n');

%% Test 2: Volatility model (gamma=0, r=2, k=0)
fprintf('Test 2: Volatility-only model (gamma=0, r=2, k=0)\n');
fprintf('-----------------------------------------------\n');

r = 2;
k = 0;

mu_vec    = [0; 0];          % same mean
sigma_vec = [0.04; 0.12];
gamma_vec = [];   % no AR
params_vol = create_params(0.8, 0.0, -0.5, mu_vec, gamma_vec, sigma_vec, 0);
theta_vec  = params_to_vector(params_vol);

expected_len = 3 + r + k + r;  % = 3 + 2 + 0 + 2 = 7
fprintf('Vector length: %d (expected: %d)\n', length(theta_vec), expected_len);
assert(length(theta_vec) == expected_len);

params_back = vector_to_params(theta_vec, k, r);
assert(all(params_back.gamma == 0));

fprintf('[OK] Test passed\n\n');

%% Test 3: AR(2) model (r=2, k=2)
fprintf('Test 3: AR(2) model (r=2, k=2)\n');
fprintf('------------------------------\n');

r = 2;
k = 2;

mu_vec    = [0.6; 3.0];
sigma_vec = [0.8; 0.8];
gamma_vec = [0.5; 0.3];

params_ar2 = create_params(0.8, 0.0, -0.5, mu_vec, gamma_vec, sigma_vec, k);
theta_vec  = params_to_vector(params_ar2);

expected_len = 3 + r + k + r; % 3 + 2 + 2 + 2 = 9
fprintf('Vector length: %d (expected: %d)\n', length(theta_vec), expected_len);
assert(length(theta_vec) == expected_len);

params_back = vector_to_params(theta_vec, k, r);

assert(params_back.k == k);
assert(isequal(params_back.gamma(:), gamma_vec(:)));

fprintf('[OK] Test passed\n\n');

%% Test 4: Parameter ordering
fprintf('Test 4: Parameter ordering\n');
fprintf('-----------------------------\n');

r = 2;
k = 1;

mu_vec    = [0.4; 0.5];   % mu(1), mu(2)
sigma_vec = [0.7; 0.8];   % sigma(1), sigma(2)
gamma_vec = 0.6;

params = create_params(0.1, 0.2, 0.3, mu_vec, gamma_vec, sigma_vec, k);
theta_vec = params_to_vector(params);

% Expected layout:
% [ alpha; tau; rho; mu1; mu2; gamma1; sigma1; sigma2 ]
assert(theta_vec(1) == 0.1);  % alpha
assert(theta_vec(2) == 0.2);  % tau
assert(theta_vec(3) == 0.3);  % rho
assert(theta_vec(4) == 0.4);  % mu(1)
assert(theta_vec(5) == 0.5);  % mu(2)
assert(theta_vec(6) == 0.6);  % gamma(1)
assert(theta_vec(7) == 0.7);  % sigma(1)
assert(theta_vec(8) == 0.8);  % sigma(2)

fprintf('[OK] Correct order:\n');
fprintf('  theta(1) = alpha   = %.1f\n', theta_vec(1));
fprintf('  theta(2) = tau     = %.1f\n', theta_vec(2));
fprintf('  theta(3) = rho     = %.1f\n', theta_vec(3));
fprintf('  theta(4) = mu(1)   = %.1f\n', theta_vec(4));
fprintf('  theta(5) = mu(2)   = %.1f\n', theta_vec(5));
fprintf('  theta(6) = gamma   = %.1f\n', theta_vec(6));
fprintf('  theta(7) = sigma(1)= %.1f\n', theta_vec(7));
fprintf('  theta(8) = sigma(2)= %.1f\n\n', theta_vec(8));

%% Test 5: ERROR – wrong vector length
fprintf('Test 5: Expected ERROR (wrong vector length)\n');
fprintf('--------------------------------------------\n');

r = 2; k = 1;
try
    theta_bad = [0.8; 0.0];  % too short
    params_bad = vector_to_params(theta_bad, k, r);
    fprintf('[KO] No error raised but one was expected!\n');
catch ME
    fprintf('[OK] Error correctly detected: %s\n\n', ME.message);
end

%% Test 6: AR(3) model (r=2, k=3)
fprintf('Test 6: AR(3) model (r=2, k=3)\n');
fprintf('------------------------------\n');

r = 2;
k = 3;

mu_vec    = [0.6; 3.0];
sigma_vec = [0.8; 0.8];
gamma_vec = [0.5; 0.3; 0.1];

params_ar3 = create_params(0.8, 0.0, -0.5, mu_vec, gamma_vec, sigma_vec, k);
theta_vec  = params_to_vector(params_ar3);

expected_len = 3 + r + k + r; % 3 + 2 + 3 + 2 = 10
fprintf('Vector length: %d (expected: %d)\n', length(theta_vec), expected_len);
assert(length(theta_vec) == expected_len);

params_back = vector_to_params(theta_vec, k, r);

assert(params_back.k == k);
assert(isequal(params_back.gamma(:), gamma_vec(:)));

fprintf('[OK] Test passed\n\n');

%% Test 7: Practical workflow (estimation)
fprintf('Test 7: Practical workflow simulation\n');
fprintf('---------------------------------------\n');

r = 2; k = 1;
mu_vec    = [0; 0];
sigma_vec = [0.05; 0.10];
gamma_vec = 0;

params_init = create_params(0.5, 0.0, 0.0, mu_vec, gamma_vec, sigma_vec, k);

theta_init = params_to_vector(params_init);
fprintf('Initial vector: ');
fprintf('%.4f ', theta_init);
fprintf('\n');

theta_optimized = theta_init + 0.05 * randn(size(theta_init));
fprintf('Optimizer result: ');
fprintf('%.4f ', theta_optimized);
fprintf('\n');

params_estimated = vector_to_params(theta_optimized, k, r);

fprintf('Estimated parameters (structure):\n');
fprintf('  alpha = %.4f\n', params_estimated.alpha);
fprintf('  rho   = %.4f\n', params_estimated.rho);

fprintf('\n[OK] Full workflow simulation successful\n\n');

%% Summary
fprintf('========================================\n');
fprintf('ALL TESTS PASSED (param conversion)\n');
fprintf('========================================\n');
