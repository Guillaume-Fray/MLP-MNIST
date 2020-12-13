% --- Model Evaluation File ---

% load training-validation data
train_val_images = loadMNISTImages('train-images-idx3-ubyte');
train_val_labels_origin = loadMNISTLabels('train-labels-idx1-ubyte');
[n,k] = size(train_val_images(:,:));
train_val_labels = zeros(k,10);

% load test data
test_inputs = loadMNISTImages('t10k-images-idx3-ubyte');
test_labels = loadMNISTLabels('t10k-labels-idx1-ubyte');


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



           % --- 1 layer with 50 neurons, > 10 Epochs and L.rate = 0.1, gives a 2.1 - 2.2% error rate ---


% Create an MLP with n=784 inputs (pixels), 3 hidden units, 10 outputs for 10 digits
m = MLP(n, 50, 10);
% Initialize weights in a range +/- 1
m.initWeights(1.0); 


tic



% Number of K-folds
% for i = 1:5

% Random Cross-Validation Step
cross_valid_num = 10000;
valid_pose_start =  randi([1 45000],1); % randomly choose the starting position of the validation set
fprintf('The starting position of the validation set is at %i in the entire dataset \n', valid_pose_start);%

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

% Create the outputs arrays
training_outputs = zeros(50000,10);
valid_outputs = zeros(10000,10);
test_outputs = zeros(10000,10);


     
% x is the number of Epochs (= number of times we pass the dataset through
% the MLP)
for x=1:10 % 10000
    % Train to output the right figures
    if mod(x,1) == 0  % 10, 100
        fprintf('x is at %i \n', x);% to see where the program is at. (cycle id)
    end
    for i = 1:num_train_input % change to k when cross-validation ready
        m.adapt_to_target(training_input(:,i), training_target(:,i), 0.1); % 0.1 %%%%%%%%%%%%%% <----- RATE
        o_train = m.compute_output(training_input(:,i));
        training_outputs(i,:) = o_train;
    end
end

% Save the trained model
trained_hidden_weights = m.hiddenWeights;
trained_output_weights = m.outputWeights;
m_test = m.set_to_trained_model(trained_hidden_weights, trained_output_weights);

% Get the test outputs
for i = 1:10000
    o_test = m_test.compute_output(test_inputs(:,i));
    test_outputs(i,:) = o_test;
end



% 
disp('----- Targets -----');
display_network(test_inputs(:,1:10000));
% disp('labels(:,1:num_input)');
% disp(labels(:,1:num_input));
% disp('----- Outputs -----');
% disp(outputs(1:num_input,:));



% Training-validation performance evaluation
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
fprintf('Number of undetermined outputs for training-validation are %i \n', undetermined);% 
fprintf('Number of correct outputs for training-validation are %i \n', correct);% 
fprintf('Number of incorrect outputs for training-validation are %i \n', incorrect);% 

error_rate = (1-(correct/num_train_input))*100;
fprintf('The training-validation error rate is %f%% \n', error_rate);%
disp(' ');



% Test performance evaluation
undetermined2 = 0; %Counter
correct2 = 0; %Counter
incorrect2 = 0; %Counter
for i = 1:10000
    determined2 = 0; %Boolean
    for j = 1:10
        if test_outputs(i,j) > 0.5 && test_outputs(i,j) == max(test_outputs(i,1:10))
            determined2 = 1;
            if j-1 == test_labels(i,1)
                correct2 = correct2 + 1;
            else
                incorrect2 = incorrect2 + 1;
            end
        elseif determined2 == 0 && j == 10
            undetermined = undetermined + 1;
        end
    end
end    

disp(' ');%
disp(' ');%
fprintf('Number of undetermined test outputs are %i \n', undetermined2);% 
fprintf('Number of correct test outputs are %i \n', correct2);% 
fprintf('Number of incorrect test outputs are %i \n', incorrect2);% 

error_rate2 = (1-(correct2/10000))*100;
fprintf('The test error rate is %f%% \n', error_rate2);%
disp(' ');



toc












