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
        if j == tget+1
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
m = MLP_b2(n, 150, 10);
% Initialize weights in a range +/- 1
m.initWeights(1.0); 
% 60,000 outputs. Each output is an array of 10 binary values
outputs = zeros(k,10);


num_input = k/2; %  --- Number of images the model is trained with ---



tic

% for x=1:1000 % 10000
%     % Train to output the right figures
%     if mod(x,100) == 0  % 10, 100
%         fprintf('x is at %i \n', x);% to see where the program is at. (cycle id)
%     end
for i = 1:num_input %k
    m.adapt_to_target(images(:,i), labels(:,i), 1.2); %%%%%%%%%%%%%%% <----- RATE
    target = labels(:,i);
    o = m.compute_output(images(:,i));
    outputs(i,:) = o;
end
% end



%%%%%%%%-------------------%%%%%%%%%%%
disp('----- Targets -----');
display_network(images(:,1:num_input));
% disp('labels(:,1:num_input)');
% disp(labels(:,1:num_input));
% disp('----- Outputs -----');
% disp(outputs(1:num_input,:));

undetermined = 0; %Counter
correct = 0; %Counter
incorrect = 0; %Counter
for i = 1:num_input
    determined = 0; %Boolean
    for j = 1:10
        if outputs(i,j) > 0.5 && outputs(i,j) == max(outputs(i,1:10))
%             fprintf('Output %i is %i \n', i, j-1);%
            determined = 1;
%             fprintf('Label %i is %i \n', i, labels_origin(i,1));%
            if j-1 == labels_origin(i,1)
                correct = correct + 1;
            else
                incorrect = incorrect + 1;
            end
        elseif determined == 0 && j == 10
%             disp(' ');%
%             fprintf('Output %i with label %i could not be determined \n', i, labels_origin(i,1));%
%             disp(' ');%
            undetermined = undetermined + 1;
        end
    end
end    

disp(' ');%
disp(' ');%
fprintf('Number of undetermined outputs are %i \n', undetermined);% 
fprintf('Number of correct outputs are %i \n', correct);% 
fprintf('Number of incorrect outputs are %i \n', incorrect);% 




toc
















