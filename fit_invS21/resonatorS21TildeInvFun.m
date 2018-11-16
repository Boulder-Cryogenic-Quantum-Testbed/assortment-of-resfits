function out = resonatorS21TildeInvFun(params,freq)
% params(1) = Qi
% params(2) = Qc*
% params(3) = phi
% params(4) = f_0

out = zeros(length(freq),2);
out(:,1) = 1 + params(1)/params(2) * ( cos(params(3))+2*params(1)*(freq-params(4))/params(4)*sin(params(3)) )./( 1+(2*params(1)*(freq-params(4))/params(4)).^2 );
out(:,2) = 0 + params(1)/params(2) * ( sin(params(3))-2*params(1)*(freq-params(4))/params(4)*cos(params(3)) )./( 1+(2*params(1)*(freq-params(4))/params(4)).^2 );

end
