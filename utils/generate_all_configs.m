function configs = generate_all_configs(k,r)
%% FUNCTION DESCRIPTION
    % Generate all state blocks configuration 
    % Indeed, the bivariate process (s_t, y_t) is a k+1-th order Markov
    % chain (cf theorem 3.1). Since y_t depends on mu_t, mu_{t-1},...,
    % mu_{t-k} (and so of s_t,...,s_{t-k}), y_{t-1},..., y_{t-k}, u_t, we
    % thus need to follow (s_t,..., s_{t-k}), which yields 2^{k+1}
    % possible combinations if there are two regimes.
    % Inputs:
    %   - k: order of the AR on y_t
    %   - r: number of regimes
    % Output:
    %   - configs: (r^{k+1} x (k+1)) matrix with all configurations. Each
    %   row is a configuration of the regime block (s_t,..., s_{t-k})
%% ========================================================================
    % Validation
    if k < 0 || k ~= floor(k)
        error('k must be an integer indicating the AR-order on y_t, but got: %.2f', k);
    end

    if r < 2 || r ~= floor(r)
        error('r must be an integer >= 2, but got: %.2f', r);
    end

    % Number of states to keep track of
    n_states = k+1;
    % Total possible configurations
    n_configs = r^n_states;

    % Generate all possible combinaitions
    configs = zeros(n_configs, n_states);

    for i=0:(n_configs - 1)
        digits = zeros(1,n_states);
        temp = i;

        for j = n_states:-1:1
            digits(j) = mod(temp,r);
            temp = floor(temp/r);
        end
        configs(i+1,:) = digits;
    end
end
