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
m = MLP(n, 3, 1);
% Initialize weights in a range +/- 1
m = m.initWeights(1.0); 
for x=1:10000
    % Train to output the right figures
    m1 = m.adapt_to_target([images(:,1)], [labels(1)], 0.8); % rate = 0.1
    o1 = m.compute_output(images(:,1));
%     m2 = m.adapt_to_target([images(:,2)], [images(:,2)], 0.8); % rate = 0.1
%     o2 = m.compute_output(images(:,2));
%     m3 = m.adapt_to_target([images(:,3)], [images(:,3)], 0.8); % rate = 0.1
%     o3 = m.compute_output(images(:,3));
%     m4 = m.adapt_to_target([images(:,4)], [images(:,4)], 0.8); % rate = 0.1
%     o4 = m.compute_output(images(:,4));
%     m5 = m.adapt_to_target([images(:,5)], [images(:,5)], 0.8); % rate = 0.1
%     o5 = m.compute_output(images(:,5));
%     m6 = m.adapt_to_target([images(:,6)], [images(:,6)], 0.8); % rate = 0.1
%     o6 = m.compute_output(images(:,6));
%     m7 = m.adapt_to_target([images(:,7)], [images(:,7)], 0.8); % rate = 0.1
%     o7 = m.compute_output(images(:,7));
%     m8 = m.adapt_to_target([images(:,8)], [images(:,8)], 0.8); % rate = 0.1
%     o8 = m.compute_output(images(:,8));
    % Outputs should approach [0 1 0]
    %[m.compute_output([0]) m.compute_output([1]) m.compute_output([2])];
end

disp('----- Targets -----');
display_network(images(:,1:8));
disp('labels(1,1)');
disp(labels(1,1));
disp('----- Outputs -----');
% disp([o1,o2,o3,o4,o5,o6,o7,o8]); 
disp(o1);
