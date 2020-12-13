% Test file
train_val_images = loadMNISTImages('train-images-idx3-ubyte');
train_val_labels_origin = loadMNISTLabels('train-labels-idx1-ubyte');
[n,k] = size(train_val_images(:,:));
train_val_labels = zeros(k,10);

% transform all 1-digit inputs into 10-binary-value inputs. e.g:
% if the original label is 5, there will be a 1 in the 5th position and 0s 
% in all the others
for i = 1:size(train_val_labels_origin(:,:))
    tget = train_val_labels_origin(i,1);
    for j = 1:10
        if j == tget+1
            train_val_labels(i,j) = 1;
        else
            train_val_labels(i,j) = 0;
        end
    end
end
train_val_labels = transpose(train_val_labels);

%%%%%%%%%%%%%%%%%------------------------ Start Modif


cross_valid_num = 10000;
valid_pose_start =  randi([1 45000],1); % randomly choose the starting position of the validation set
fprintf('The starting position of the validation set is at %i in the entire set \n', valid_pose_start);%

%  --- Images and labels the model validates the model with ---
valid_input = train_val_images(:, valid_pose_start:(valid_pose_start+9999));
valid_target = train_val_labels(:, valid_pose_start:(valid_pose_start+9999));
% disp(size(valid_input)); % [784, 10000]
% disp(size(valid_target)); % [10, 10000]


%  --- Images and labels the model is trained with ---
training_in1 = train_val_images(:, 1:(valid_pose_start-1));
training_in1 = transpose(training_in1);
training_in2 = train_val_images(:, (valid_pose_start+10000):k);
training_in2 = transpose(training_in2);
training_input = [training_in1; training_in2];
training_input = transpose(training_input);
% disp(size(training_input)); % [784, 50000]

num_train_input = 50000;
num_valid_input = 10000;

training_tar1 = train_val_labels(:, 1:(valid_pose_start-1));
training_tar1 = transpose(training_tar1);
training_tar2 = train_val_labels(:, (valid_pose_start+10000):k);
training_tar2 = transpose(training_tar2);
training_target = [training_tar1; training_tar2];
training_target = transpose(training_target);
% disp(size(training_target)); % [10, 50000]

% 60,000 outputs. Each output is an array of 10 binary values
training_outputs = zeros(50000,10);
valid_outputs = zeros(10000,10);

%%%%%%%%%%%%%%%%%------------------------ End Modif




  % --- 1 layer with 50 neurons, > 10 Epochs and L.rate = 0.1, gives a 2.1 - 2.2% error rate ---

     


% Create an MLP with n=784 inputs (pixels), 3 hidden units, 10 outputs for 10 digits
m = MLP(n, 50, 10);
% Initialize weights in a range +/- 1
m.initWeights(1.0); 



tic


% x is the number of Epochs (= number of times we pass the dataset through
% the MLP)
for x=1:10 % 10000
    % Train to output the right figures
    if mod(x,1) == 0  % 10, 100
        fprintf('x is at %i \n', x);% to see where the program is at. (cycle id)
    end
    for i = 1:num_train_input % change to k when cross-validation ready
        m.adapt_to_target(train_val_images(:,i), train_val_labels(:,i), 0.1); % 0.1 %%%%%%%%%%%%%% <----- RATE
        target = train_val_labels(:,i);
        o = m.compute_output(train_val_images(:,i));
        training_outputs(i,:) = o;
    end
end



%%%%%%%%-------------------%%%%%%%%%%%
disp('----- Targets -----');
display_network(train_val_images(:,1:num_train_input));
% disp('labels(:,1:num_input)');
% disp(labels(:,1:num_input));
% disp('----- Outputs -----');
% disp(outputs(1:num_input,:));

undetermined = 0; %Counter
correct = 0; %Counter
incorrect = 0; %Counter
for i = 1:num_train_input
    determined = 0; %Boolean
    for j = 1:10
        if training_outputs(i,j) > 0.5 && training_outputs(i,j) == max(training_outputs(i,1:10))
%             fprintf('Output %i is %i \n', i, j-1);%
            determined = 1;
%             fprintf('Label %i is %i \n', i, labels_origin(i,1));%
            if j-1 == train_val_labels_origin(i,1)
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

error_rate = 1-(correct/num_train_input);
fprintf('The error rate is %f %% \n', error_rate);%
disp(' ');


toc
















