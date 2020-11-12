classdef MLP < handle
    properties (SetAccess=private)
        inputDim  % number of inputs/features x (1st vector)
        hidden1Dim   % number of neurons in the unique hidden layer (No 1)
                     % --> unique for now
        outputDim  % number of output nodes
        
        hiddenWeights  % weights before the hidden layer
        outputWeights  % weights after the hidden layer
    end
    
    methods
        % Build the Multilayer Perceptron object
        function obj = MLP(inputD, hiddenD, outputD)
            obj.inputDim = inputD;
            obj.hidden1Dim = hiddenD;
            obj.outputDim = outputD;
            
            obj.hiddenWeights = zeros(hiddenD, inputD+1);
            obj.outputWeights = zeros(outputD, hiddenD+1);
        end
        
        % Initialise the weights (Matrix) in a randomly manner after taking
        % the perceptron object and its variance as arguments.
        % --> try variance = 1.0    (with 0 for mean)
        function obj = initWeights(obj, variance)
            [neurons1, inputs] = size(obj.hiddenWeights);
            [outputs, neurons1_plus1] = size(obj.hiddenWeights);
            
            obj.hiddenWeights = variance.*randn([neurons1 inputs]); % +0
            obj.outputWeights = variance.*randn([outputs neurons1_plus1]); % +0
            
            disp('\nhidden weights: ');
            disp(obj.hiddenWeights(:,:));
            disp('output weights: ');
            disp(obj.outputWeights(:,:));
        
        end
        
        function [hiddenNet, hidden, outputNet, output] = compute_net_activation(obj, input)
            %hiddenNet = % TODO
            %hidden = % TODO
            %outputNet = % TODO
            %output = % TODO
        end
        
        function output = compute_output(obj,input)
            [hN, h, oN,output] = obj.compute_net_activation(input);
        end
        
        function obj = adapt_to_target(obj, input, target, rate)
            [hN, h, oN, o] = obj.compute_net_activation(input);
            
            % TODO 
        end
    end
end
