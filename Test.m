% Test file
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
[n,k] = size(images(:,:));

% Show the first image
% display_network(images(:,1));
% disp(labels(1));

% Show the first 8 images
% display_network(images(:,1:8));
% disp(labels(1:8));


% Create an MLP with 1 input, 3 hidden units, 1 output
m = MLP(n, 3, 10);
% Initialize weights in a range +/- 1
m.initWeights(1.0); 
output = zeros(1,10);%%%%%%%%%%%%%%

disp('output');
disp(output);

for i=1 % 1:8, length(labels)
    for x=1:10  % 100, 1000
        % Train to output the right figures
        m.adapt_to_target(images(:,i), labels(i), 0.8); % rate = 0.1
        output(1,i) = m.compute_output(images(:,i));
    end
    display_network(images(:,i));
    fprintf('label %d : \n',i);
    disp(labels(i));
end
    % Outputs should approach [0 1 0]
    %[m.compute_output([0]) m.compute_output([1]) m.compute_output([2])];

model_output_weights = m.outputWeights;
model_hidden_weights = m.hiddenWeights;
testing = m.compute_output(images(:, 67));


%%%%%%%%-------------------%%%%%%%%%%%
disp('----- Targets -----');
display_network(images(:,1:50));
% disp('labels(1:50)');
% disp(labels(1:50));
disp('----- Outputs -----');
disp(testing);
