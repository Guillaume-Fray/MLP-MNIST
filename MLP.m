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
            [outputs, neurons1_plus1] = size(obj.outputWeights);  
%             obj.hiddenWeights = [6,0,-2; 2,-2,0];
%             obj.outputWeights = [-4,2,2];
            obj.hiddenWeights = variance.*randn([neurons1 inputs]); % +0
            obj.outputWeights = variance.*randn([outputs neurons1_plus1]); % +0
            
%             disp('original hidden weights: ');
%             disp(obj.hiddenWeights(:,:));
%             disp('original output weights: ');
%             disp(obj.outputWeights(:,:));
        
        end
        
        % This function calls the forward propagation and extracts all
        % outputs.
        function [hiddenNet, hidden, outputNet, output] = compute_net_activation(obj, input)
            % Set vector v(I1; I2; ... ; 1)
            input = [input; 1];
            % Neuron Network's dimensions: (row = nb of Neurons, column = 1
            % layer)
            hiddenNet = zeros(obj.hiddenDim, 1); % activation a for each N
            hidden = zeros(obj.hiddenDim, 1); % output o for each N
 
%             disp('inputs: ');
%             disp([m, n]);  
%             disp('hidden weights: ');
%             disp(obj.hiddenWeights(:,:));            
            % Net activation operation before hidden layer
            for i=1:obj.hiddenDim
                hiddenNet(i,1) = dot(obj.hiddenWeights(i, :), input);
            end
%             disp('hiddenNet: ');
%             disp(hiddenNet(:,:));
            
            % Hidden layer function (o1) operation
            for i=1:obj.hiddenDim
                hidden(i,1) = sigmoid(hiddenNet(i,1));
            end
%             disp('hidden: ');
%             disp(hidden(:,:));            
            
            
%             disp('output weights: ');
%             disp(obj.outputWeights(:,:));
            % Net activation operation after hidden layer
            outputNet = dot(obj.outputWeights, [hidden; 1]);
%             disp('outputNet: ');
%             disp(outputNet(:,:));
            
            % Output layer function (o2) operation
            output = sigmoid(outputNet);
%             disp('Final output: ');
%             disp(output(:,:));
        end
        
        % This function calls the forward propagation and extracts only the
        % overall output. It does not have to be altered.
        function output = compute_output(obj,input)
            [hN, h, oN,output] = obj.compute_net_activation(input);
        end
        
        % Back propagation of errors: learning algorithm in this method
        function obj = adapt_to_target(obj, input, target, rate)
            [hN, h, oN, o] = obj.compute_net_activation(input);
            
            dEo = o - target; % 
            dOa = o.*(ones(length(o),1)-o); % 
            d2 = dEo.*dOa; % matrix
%             disp('dEo: ');
%             disp(dEo(:,:));                 
%             disp('dOa: ');
%             disp(dOa(:,:));             
%             disp('d2: ');
%             disp((d2(:,:)));
%             disp('outputWeights: ');
%             disp(obj.outputWeights(:,:));            
%             disp('h: ');
%             disp(h(:,:));             
            delta_w2 = d2*transpose([h;1]); % matrix
            disp('delta_w2: ');
            disp(delta_w2(:,:));         
          

            d_inter = d2*obj.outputWeights(1:obj.hiddenDim); %
%             disp('Hidden Weights: ');
%             disp(size(obj.hiddenWeights(:,:))); 
%             disp('d_inter: ');
%             disp(size(d_inter(:,:)));       
%             disp('h length: ');
%             disp(length(h));            
            dHa = h(1:obj.hiddenDim).*(ones(length(h),1)-h(1:obj.hiddenDim)); %
%             disp('dHa: ');
%             disp(size(dHa(:,:)));   
            d1 = transpose(d_inter).*dHa; %  
            disp('d1: ');
            disp((d1(:,:)));
            disp('input: ');
            disp((input(:,:)));  
            
            delta_w1 = d1*transpose([input;1]); %       
            disp('delta_w1: ');
            disp((delta_w1(:,:)));               
            
            obj.outputWeights = obj.outputWeights - (rate*delta_w2);
%             disp('Update outputWeights: ');
%             disp(obj.outputWeights(:,:));                       
            obj.hiddenWeights = obj.hiddenWeights - (rate*delta_w1);
%             disp('Update hiddenWeights: ');
%             disp(obj.hiddenWeights(:,:));
            disp('Final Output: ');
            disp(o);
        end
    end
end
