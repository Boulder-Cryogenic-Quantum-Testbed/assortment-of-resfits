function [ logmag ] = resonatorS21TildeLogMag( params, freq )

a = resonatorS21TildeInvFun(params,freq);
b = 1./(a(1)+1i*a(2));
logmag = 20*log10(abs(b));

end
