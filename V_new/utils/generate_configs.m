function configs = generate_configs(k)
%% FUNCTION DESCRIPTION
% Generate all binary configurations for k+1 states
% Inputs:
%   k : AR order (number of lags)
%
% Outputs:
%   configs : Matrix of size (2^(k+1) x k+1)
%             Each row is a binary configuration [s_t, s_{t-1}, ..., s_{t-k}]
%% ========================================================================

    num_states = k + 1;
    num_rows   = 2^num_states;
    configs    = zeros(num_rows, num_states);
    
    for i = 0:num_rows-1
        % Convert number to binary array (string to double vector)
        str = dec2bin(i, num_states);
        configs(i+1, :) = str - '0';
    end
end