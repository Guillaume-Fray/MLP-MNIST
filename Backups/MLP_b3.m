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
        % to zero.
        function obj = MLP(inputD, hiddenD, outputD)
            obj.inputDim = inputD;
            obj.hiddenDim = hiddenD;
            obj.outputDim = outputD;
            obj.hiddenWeights = zeros(hiddenD, inputD + 1);
            obj.outputWeights = zeros(outputD, hiddenD + 1);
        end
        
        % Initialise the weights (Matrix) in a randomly manner after taking
        % the perceptron object and its variance as arguments.
        % --> try variance = 1.0    (with 0 as mean)
        function obj = initWeights(obj, variance) 
%             obj.hiddenWeights = [6,0,-2; 2,-2,0];
%             obj.outputWeights = [-4,2,2];
%             obj.hiddenWeights = [2,-3,1; 0,1,-4];
%             obj.outputWeights = [1,0,3];
            obj.hiddenWeights = variance * randn(obj.hiddenDim, obj.inputDim + 1); % +0
            obj.outputWeights = variance * randn(obj.outputDim, obj.hiddenDim + 1); % +0
            
%             disp('original hidden weights: ');
%             disp(obj.hiddenWeights(:,:));
%             disp('original output weights: ');
%             disp(obj.outputWeights(:,:));
        
        end
 
      % Set weights to those of trained model, for actual model evaluation
        function obj = set_to_trained_model(obj,hW, oW) 
            obj.hiddenWeights = hW; % +0
            obj.outputWeights = oW; % +0           
        end        
        
        
        
        % This function calls the forward propagation and extracts all
        % outputs.
        function [hiddenNet, hidden, outputNet, output] = compute_net_activation(obj, input)

            input = [input; 1];   
            hiddenNet = obj.hiddenWeights * input;
            hidden = sigmoid(hiddenNet); 
            
            hid = [hidden; 1];           
            outputNet = obj.outputWeights * hid;
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
            
%             disp('Target: ');
%             disp(target);
%             disp('o: ');
%             disp(o); 
            dEo = o - target; %
%             disp('dEo: ');
%             disp(dEo);                
            dOa = o.*(1-o); %   
%             disp('dOa: ');
%             disp(dOa);               
            d2 = dEo.*dOa; % 
%             disp('d2: ');
%             disp(transpose(d2));                

       
            delta_w2 = d2*transpose([h;1]); %
     
            
            d_inter = transpose(obj.outputWeights) * d2; %        
            dHa = h.*(ones(length(h),1) - h); %          
            d1 = d_inter(1:length(h)).*dHa; %
%             disp('d1: ');
%             disp(d1);                
            
            delta_w1 = d1*transpose([input;1]); % 
%             disp('delta_w2: ');
%             disp(delta_w2);              
%             disp('delta_w1: ');
%             disp((delta_w1));  

%             disp('outputWeights: ');
%             disp(obj.outputWeights);
            obj.outputWeights = obj.outputWeights - (rate*delta_w2);                    
            obj.hiddenWeights = obj.hiddenWeights - (rate*delta_w1);
            
        end
    end
end
