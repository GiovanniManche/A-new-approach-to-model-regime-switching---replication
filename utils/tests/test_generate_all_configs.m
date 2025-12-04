function test_generate_all_configs()
% TEST_GENERATE_ALL_CONFIGS
%   Quick checks for generate_all_configs(k, r)

    clc; fprintf("==== TEST generate_all_configs ====\n\n");

    %% TEST 1: Simple binary case
    k = 1;  % AR(1) => states = (s_t, s_{t-1})
    r = 2;  % 2 regimes

    configs = generate_all_configs(k, r);

    fprintf("Test 1: k=1, r=2 (Expect 4x2 matrix)\n");
    disp(configs);

    if size(configs,1) == 4 && size(configs,2) == 2
        fprintf("[OK] Correct dimensions: 4 x 2\n\n");
    else
        fprintf("[FAIL] Incorrect dimensions\n\n");
    end

    %% TEST 2: Ternary case (3 regimes)
    k = 2;  % AR(2) => (s_t, s_{t-1}, s_{t-2})
    r = 3;  % 3 regimes

    configs = generate_all_configs(k, r);

    fprintf("Test 2: k=2, r=3 (Expect 27x3 matrix)\n");
    disp(configs); 

    if size(configs,1) == 27 && size(configs,2) == 3
        fprintf("[OK] Correct dimensions: 27 x 3\n\n");
    else
        fprintf("[FAIL] Incorrect dimensions\n\n");
    end

    %% TEST 3: Check values are in {1,...,r}
    if all(configs(:) >= 1) && all(configs(:) <= r)
        fprintf("[OK] Values are in {1,...,r}\n\n");
    else
        fprintf("[FAIL] Values out of range\n\n");
    end

    %% TEST 4: Check ordering
    fprintf("Test 4: Simple order verification...\n");
    fprintf("First rows:\n");
    disp(configs(1:5,:));
    fprintf("Last rows:\n");
    disp(configs(end-4:end,:));

    fprintf("\nIf the last row = (r, ..., r), order is consistent.\n");
    fprintf("\n==== END TEST ====\n");
end
