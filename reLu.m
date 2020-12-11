function [o] = reLu(a)
%% --- Function o = f(a) ---
% Defined as the Rectified Linear Unit function in this case here.
  o = zeros(size(a),1);
  for i = 1:size(a)
      if a(i)<0
          o(i) = 0;
      else
          o(i) = a;
      end
  end
end