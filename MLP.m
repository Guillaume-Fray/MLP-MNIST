classdef MLP < handle
    properties (SetAccess=private)
        inputDim
        hiddenDim
        outputDim
        
        hiddenWeights
        outputWeights
    end
    
    methods
        function obj=MLP(inputD,hiddenD,outputD)
            obj.inputDim=inputD;
            obj.hiddenDim=hiddenD;
            obj.outputDim=outputD;
            obj.hiddenWeights=zeros(hiddenD,inputD+1);
            obj.outputWeights=zeros(outputD,hiddenD+1);
        end
        
        function obj=initWeight(obj,variance)
            %obj.hiddenWeights=% TODO
            %obj.outputWeights=% TODO
        end
        
        function [hiddenNet,hidden,outputNet,output]= ...
            compute_net_activation(obj, input)
            %hiddenNet = % TODO
            %hidden = % TODO
            %outputNet = % TODO
            %output = % TODO
        end
        
        function output=compute_output(obj,input)
            [hN,h,oN,output] = obj.compute_net_activation(input);
        end
        
        function obj=adapt_to_target(obj,input,target,rate)
            [hN,h,oN,o] = obj.compute_net_activation(input);
            
            % TODO 
        end
    end
end
