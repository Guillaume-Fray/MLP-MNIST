classdef MLP < handle
    properties (SetAccess=private)
        inputDim % Number of inputs
        hiddenDim % Number of hidden neurons
        outputDim % Number of outputs     
        hiddenWeights % Weight matrix for the hidden layer, format (hiddenDim)x(inputDim+1) to include bias terms
        outputWeights % Weight matrix for the output layer, format (outputDim)x(hiddenDim+1) to include bias terms
    end
    
    methods
        % Constructor: Initialize to given dimensions and set all weights
        % zero.
        function obj = MLP(inputD, hiddenD, outputD)
            obj.inputDim = inputD;
            obj.hiddenDim = hiddenD;
            obj.outputDim = outputD;
            obj.hiddenWeights = zeros(hiddenD, inputD+1);
            obj.outputWeights = zeros(outputD, hiddenD+1);
        end
        
        % Initialise the weights (Matrix) in a randomly manner after taking
        % the perceptron object and its variance as arguments.
        % --> try variance = 1.0    (with 0 as mean)
        function obj = initWeights(obj, variance)
            [neurons1, inputs] = size(obj.hiddenWeights);
            [outputs, neurons1_plus1] = size(obj.hiddenWeights);     
            obj.hiddenWeights = variance.*randn([neurons1 inputs]); % +0
            obj.outputWeights = variance.*randn([outputs neurons1_plus1]); % +0
            
            disp('hidden weights: ');
            disp(obj.hiddenWeights(:,:));
            disp('output weights: ');
            disp(obj.outputWeights(:,:));
        
        end
        
        % This function calls the forward propagation and extracts all
        % outputs.
        function [hiddenNet, hidden, outputNet, output] = compute_net_activation(obj, input)
            hiddenNet = dot(input, obj.hiddenWeights); % Net activation operation before hidden layer
            hidden = sigmoid(hiddenNet); % Hidden layer function (o1) operation
            outputNet = dot(hidden, obj.outputWeights); % Net activation operation after hidden layer
            output = outputXOr(outputNet); % Output layer function (o2) operation
        end
        
        % This function calls the forward propagation and extracts only the
        % overall output. It does not have to be altered.
        function output = compute_output(obj,input)
            [hN, h, oN,output] = obj.compute_net_activation(input);
        end
        
        % Backward-propagation of errors: learning algorithm in this method
        function obj = adapt_to_target(obj, input, target, rate)
            [hN, h, oN, o] = obj.compute_net_activation(input);
            if o - target < 0  % means target undershot
                % 
            elseif o - target > 0  % means target overshot
                
            else  % o = target. Unlikely     
                
            % TODO 
        end
    end
end
