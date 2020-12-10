% Create an MLP with 1 input, 3 hidden units, 1 output
m = MLP(1, 3, 1);
% Initialize weights in a range +/- 1
m = m.initWeights(1.0); 
for x=1:10000
    % Train to output [0],[1],[0] for inputs [0],[1],[2]
    m1 = m.adapt_to_target([0], [0], 0.8); % rate = 0.1
    m2 = m.adapt_to_target([1], [1], 0.8); % rate = 0.1
    m3 = m.adapt_to_target([2], [0], 0.8); % rate = 0.1
    % Outputs should approach [0 1 0]
    [m.compute_output([0]) m.compute_output([1]) m.compute_output([2])];
end

disp('----- Outputs -----');
disp([m1.compute_output([0]) m2.compute_output([1]) m3.compute_output([2])]);     
