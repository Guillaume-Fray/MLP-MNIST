% A Multi-layer perceptron class
classdef MLP_backup < handle
    % Member data
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
        function obj=MLP(inputD,hiddenD,outputD)
            obj.inputDim=inputD;
            obj.hiddenDim=hiddenD;
            obj.outputDim=outputD;
            obj.hiddenWeights=zeros(hiddenD,inputD+1);
            obj.outputWeights=zeros(outputD,hiddenD+1);
        end
        
        % TODO Implement a randomized initialization of the weight
        % matrices.
        % Use the 'variance' parameter to control the spread of initial
        % values.
        function obj=initWeight(obj,variance)
            % Note: 'obj' here takes the role of 'this' (Java/C++) or
            % 'self' (Python), refering to the object instance this member
            % function is run on.
            %obj.hiddenWeights=% TODO
            %obj.outputWeights=% TODO
        end
        
        % TODO Implement the forward-propagation of values algorithm in
        % this method.
        % hiddenNet ~ net activation of the hidden-layer neurons
        % hidden ~ output of the hidden-layer neurons
        % outputNet ~ net activation of the output-layer neurons
        % output ~ output of the output-layer neurons
        % Note: the return value is automatically fit into a array
        % containing the above four elements
        function [hiddenNet,hidden,outputNet,output]=compute_net_activation(obj, input)
            %hiddenNet = % TODO
            %hidden = % TODO
            %outputNet = % TODO
            %output = % TODO
        end
        
        % This function calls the forward propagation and extracts only the
        % overall output. It does not have to be altered.
        function output=compute_output(obj,input)
            [hN,h,oN,output] = obj.compute_net_activation(input);
        end
        
        % TODO Implement the backward-propagation of errors (learning) algorithm in
        % this method.
        function obj=adapt_to_target(obj,input,target,rate)
            [hN,h,oN,o] = obj.compute_net_activation(input);
            
            % TODO 
        end
    end
end
