function [o] = reLu(a)
%% --- Function o = f(a) ---
% Defined as the Rectified Linear Unit function in this case here.
  if a<0
      o = 0;
  else
      o = a;
  end
end