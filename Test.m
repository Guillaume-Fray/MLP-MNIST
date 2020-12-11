% Test file
images = loadMNISTImages('train-images-idx3-ubyte');
labels_origin = loadMNISTLabels('train-labels-idx1-ubyte');
[n,k] = size(images(:,:));
labels = zeros(k,10);

% transform all 1-digit inputs into 10-binary-value inputs. e.g:
% if the original label is 5, there will be a 1 in the 5th position and 0s 
% in all the others
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


% Create an MLP with n=784 inputs (pixels), 3 hidden units, 10 outputs for 10 digits
m = MLP(n, 3, 10);
% Initialize weights in a range +/- 1
m.initWeights(1.0); 
% 60,000 outputs. Each output is an array of 10 binary values
outputs = zeros(k,10);

tic

for x=1:10000 % 10000
    % Train to output the right figures
    if mod(x,1000) == 0
        fprintf('x is at %i \n', x);% 
    end
    for i = 1:50 %k
        if mod(i,1000) == 0
            fprintf('           image --> %i \n', i);% 
        end  
        m.adapt_to_target(images(:,i), labels(:,i), 1.7); %%%%%%%%%%%%%%%
        target = labels(:,i);
        o = m.compute_output(images(:,i));
        outputs(i,:) = o;
    end
end



%%%%%%%%-------------------%%%%%%%%%%%
disp('----- Targets -----');
display_network(images(:,1:50));
disp('labels(:,1:50)');
disp(labels(:,1:50));
disp('----- Outputs -----');
disp(outputs(1:50,:));

undetermined = 0; %Counter
for i = 1:50
    determined = 0; %Boolean
    for j = 1:10
        if outputs(i,j) > 0.9
            fprintf('Output %i is %i \n', i, j);%
            determined = 1;
        elseif determined == 0 && j == 10
            fprintf('Output %i could not be determined \n', i);% 
            undetermined = undetermined + 1;
        end
    end
end    

fprintf('Number of undetermined outputs are %i \n', undetermined);% 


toc








