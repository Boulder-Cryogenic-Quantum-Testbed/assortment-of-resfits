function [ photons ] = power2photons_hanger( Qr, Papp, Patt, f0 )
% This function computes the number of photons in the resonator given total quality factor Qr,
% applied power Papp (dB), attenuation in fridge Patt (dB), and resonator frequency (GHz).
% Adapted from Junling's Mathematica sheet.

hbar = 1.0545718E-34;

w0 = 2*pi*f0*10^(9);
Pt = 10.^((Papp + Patt -30)/10); % Power in the transmission line near the resonator, in Watts
Pin = (4/9)*Pt; % Fraction of power in the resonator depends on geometry. 4/9 corresponds to the hanger geometry. A factor of 1 corresponds to reflection geometry.
kappa = w0./Qr;
photons = Pin./(hbar*w0.*kappa);

end

