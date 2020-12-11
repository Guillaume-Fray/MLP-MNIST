% Test file ---> WORKS for several images!!!! Finally...
images = loadMNISTImages('train-images-idx3-ubyte');
labels_origin = loadMNISTLabels('train-labels-idx1-ubyte');
[n,k] = size(images(:,:));
labels = zeros(k,10);

for i = 1:size(labels_origin(:,:))
    tget = labels_origin(i,1);
    for j = 1:10
        if j == tget
            labels(i,j) = 1;
        else
            labels(i,j) = 0;
        end
    end
end
labels = transpose(labels);



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
 
outputs = zeros(k,10);


for x=1:10000 % 10000
    % Train to output the right figures
    for i = 1:10 %k
        m.adapt_to_target(images(:,i), labels(:,i), 0.8); % best, 0.8, rate = 0.1
        target = labels(:,i);
        o = m.compute_output(images(:,i));
        outputs(i,:) = o;
    end
end



%%%%%%%%-------------------%%%%%%%%%%%
disp('----- Targets -----');
display_network(images(:,1:50));
disp('labels(:,1)');
disp(labels(:,1));
disp('----- Outputs -----');
disp(outputs(1:8,:));
