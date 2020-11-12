function [o] = sigmoid_o(a)
%% --- Function o = f(a) ---
% Defined as the Sigmoid function in this case here.
o = 1/(1+(exp(-a)));
end