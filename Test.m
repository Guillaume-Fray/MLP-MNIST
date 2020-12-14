% --- Model Evaluation File ---
clear all;
close all;

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



           % --- 1 layer with 50 neurons, >= 10 Epochs and L.rate = 0.1 ---


% Create an MLP with n=784 inputs (pixels), 3 hidden units, 10 outputs for 10 digits
m = MLP(n, 50, 10);  %%% 50

% Create the outputs arrays
training_outputs = zeros(50000,10);
valid_outputs = zeros(10000,10);
test_outputs = zeros(10000,10);
folds_errors = zeros(6,1);
total_error = 0;


tic



% Number of K-folds
for folds = 1:6
    
    % Randomly (re-)initializes weights in a range +/- 1
    % Basically discards the previous model and what it had learnt
    m.initWeights(1.0); 
    
    % Random Cross-Validation Step
    cross_valid_num = 10000;
    valid_pose_start =  1+((folds-1)*10000); % randomly choose the starting position of the validation set
    fprintf('The starting position of the validation set for fold %i is at %i in the entire dataset \n', folds, valid_pose_start);%

    % Images and labels the model validates the model with
    valid_input = train_val_images(:, valid_pose_start:(valid_pose_start+9999));
    valid_target = train_val_labels(:, valid_pose_start:(valid_pose_start+9999));
    % disp(size(valid_input)); % [784, 10000]
    % disp(size(valid_target)); % [10, 10000]


    % Images and labels the model is trained with
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
    

    % x is the number of Epochs (= number of times we pass the dataset 
    % through the MLP to make it learn deeper after every iteration)
    for x=1:10 % 10000
        %tic
        if mod(x,1) == 0  % 5, 10
            fprintf('Epoch %i \n', x);% to see where the program is at. (Epoch ID)
        end
        % Train to output the right figures        
        for i = 1:num_train_input % 
            m.adapt_to_target(training_input(:,i), training_target(:,i), 0.3); % 0.1 %%%%%%%%%%%%%% <----- L. RATE
            o_train = m.compute_output(training_input(:,i));
            training_outputs(i,:) = o_train;
        end
        %toc
    end
    
    % Keep the trained model for validation
    trained_hidden_weights = m.hiddenWeights;
    trained_output_weights = m.outputWeights;
    m_valid = m.set_to_trained_model(trained_hidden_weights, trained_output_weights);

    % Get the valid outputs
    for i = 1:10000
        o_valid = m_valid.compute_output(valid_input(:,i));
        valid_outputs(i,:) = o_valid;
    end
    
    
    % Training-validation performance evaluation
    undetermined = 0; %Counter
    correct = 0; %Counter
    incorrect = 0; %Counter
    for i = 1:10000
        determined = 0; %Boolean
        for j = 1:10
            if valid_outputs(i,j) > 0.5 && valid_outputs(i,j) == max(valid_outputs(i,1:10))
                determined = 1;
%                 fprintf('Train_val_labels_origin %i is %i  \n', i, train_val_labels_origin(valid_pose_start+(i-1),1));%
%                 fprintf('valid_output %i is %i  \n', i, (j-1));%
%                 disp(' ');
                if j-1 == train_val_labels_origin(valid_pose_start+(i-1),1)
                    correct = correct + 1;
                else
                    incorrect = incorrect + 1;
                end
            elseif determined == 0 && j == 10
                undetermined = undetermined + 1;
            end
        end
    end  
    disp(' ');%
    disp(' ');%
    fprintf('Number of undetermined outputs for training-validation for fold %i are %i \n',folds, undetermined);% 
    fprintf('Number of correct outputs for training-validation for fold %i are %i \n',folds, correct);% 
    fprintf('Number of incorrect outputs for training-validation for fold %i are %i \n',folds, incorrect);% 
    error = (1-(correct/10000))*100;
    fprintf('The training-validation error rate for fold %i is %f%% \n', folds, error);%
    disp(' ');
    folds_errors(folds,1) = error;
    total_error = total_error + error;
    
    if folds==1
        best_hidden_weights = m_valid.hiddenWeights;
        best_output_weights = m_valid.outputWeights;
    end
    
    % keep the best model after the 6 k-folds validation runs
    if folds>1 && error<folds_errors((folds-1),1)
        best_hidden_weights = m_valid.hiddenWeights;
        best_output_weights = m_valid.outputWeights;
    end
    
end    
    
average_validation_error = total_error/6;
disp(' ');
fprintf('The average training-validation error rate is %f%% \n', average_validation_error);%
disp(' ');   


% Keep the best trained model
m_test = m.set_to_trained_model(best_hidden_weights, best_output_weights);

% Get the test outputs
for i = 1:10000
    o_test = m_test.compute_output(test_inputs(:,i));
    test_outputs(i,:) = o_test;
end


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
            undetermined2 = undetermined2 + 1;
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


% END
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
disp('----- Targets -----');
display_network(test_inputs(:,1:10000));
% disp('labels(:,1:num_input)');
% disp(labels(:,1:num_input));
% disp('----- Outputs -----');
% disp(outputs(1:num_input,:));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 





toc












