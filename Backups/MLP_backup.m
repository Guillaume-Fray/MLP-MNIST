classdef MLP_backup < handle
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
        
        % set weights to those of the trained model
        function obj = chooseWeights(obj, hW, oW) 
            obj.hiddenWeights = hW;
            obj.outputWeights = oW;                
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
        
            % Net activation operation before hidden layer
            for i=1:obj.hiddenDim
                hiddenNet(i,1) = obj.hiddenWeights(i, :) * input;
            end

            % Hidden layer function (o1) operation
            for i=1:obj.hiddenDim
                hidden(i,1) = sigmoid(hiddenNet(i,1));
            end

            % Net activation operation after hidden layer
            outputNet = obj.outputWeights * [hidden; 1];
            % Output layer function (o2) operation
            output = sigmoid(outputNet);

        end
        
        % This function calls the forward propagation and extracts only the
        % overall output. It does not have to be altered.
        function output = compute_output(obj,input)
            [hN, h, oN,output] = obj.compute_net_activation(input);
        end
        
        % Back propagation of errors: learning algorithm in this method
        function obj = adapt_to_target(obj, input, target, rate)
            [hN, h, oN, o] = obj.compute_net_activation(input);
            
            disp('Target: ');
            disp(target);            
            dEo = o - target; % 
            dOa = o.*(ones(1,length(o))-o); %        
            d2 = dEo.*dOa; % 
            d2 = transpose(d2);                  
            delta_w2 = d2*transpose([h;1]); %
%             disp('delta_w2: ');             
%             disp((delta_w2(:,:)));  


%             disp('d2: ');             
%             disp((d2(:,:)));
%             disp('h: ');             
%             disp((h(:,:)));          
            d_inter = transpose(obj.outputWeights) * d2; %        
            dHa = h.*(ones(length(h),1) - h); %
%             disp('d_inter: ');
%             disp((d_inter(:,:)));
%             disp('dHa: ');
%             disp((dHa(:,:)));
            
%             disp('d_inter - 1: ');
%             disp(d_inter(1:length(h)));           
            d1 = d_inter(1:length(h)).*dHa; %  
%             disp('d1: ');
%             disp((d1(:,:)));
%             disp('input: ');
%             disp(size([input;1]));            
            delta_w1 = d1*transpose([input;1]); % 
%             disp('delta_w2: ');
%             disp(delta_w2);              
%             disp('delta_w1: ');
%             disp(delta_w1);  


            obj.outputWeights = obj.outputWeights - (rate*delta_w2);                    
            obj.hiddenWeights = obj.hiddenWeights - (rate*delta_w1);
            

            disp('Final Output: ');
            disp(o);
        end
    end
end
