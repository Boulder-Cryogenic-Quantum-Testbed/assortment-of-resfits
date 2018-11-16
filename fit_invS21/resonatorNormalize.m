function [ data_freq,S21_tilde,S21_logm,S21_logm_norm,S21_phase,S21_phase_norm,p_logm,p_phase ] = resonatorNormalize(freq,S21mag,S21ph,opt)
%RESONATORNORMALIZE Normalize the S21 data for a resonator
%   The normalization is done in the complex plane using off resonance data
%   points. Outputs the resulting S-Parameters as complex data.
%
%   Required argument:
%   data: S-Parameter data to normalize, Nx3 (S21 only) or Nx9 matrix
%   Optional argument:
%   a) npts: Number of points to take at the extermities of the given
%            S-Parameter data if no explicit normalization data is given.
%            If no optional argument is given, this will default to 10 pts.
%   b) norm: Data points outside the resonator frequency to which a line
%            will be fit for normalization.
if nargin==1; opt = 10; end % Default number of points to take at extremities
VERSION = 1;

data = [freq, S21mag, S21ph];

% Check inputs
if size(opt,2)==1 % Take normalization data from extremities of data
    norm = [data(1:opt,:);data(end-opt+1:end,:)];
elseif size(opt,2)~=1 % Given normalization data
    norm = opt;
end
% Support for full S2P data arrays (9 cols), or just S21 arrays (3 cols)
if size(data,2)==9&&size(norm,2)==9
    LOGMCOL = 4;
    PHASECOL = 5;
elseif size(data,2)==3&&size(norm,2)==3
    LOGMCOL = 2;
    PHASECOL = 3;
else
    error('Unrecognized array format!')
end

datapts = size(data,1);
normpts = size(norm,1);

% Separate data in clear variables and unwrap phase together
data_freq  = data(:,1);
norm_freq  = norm(:,1);
S21_logm   = data(:,LOGMCOL);
norm_logm  = norm(:,LOGMCOL);
phase_dat  = unwrap([norm(1:normpts/2,PHASECOL); data(:,PHASECOL); norm((1+normpts/2):end,PHASECOL)]*pi/180);
S21_phase  = phase_dat(normpts/2+1:normpts/2+datapts);
norm_phase = [phase_dat(1:normpts/2); phase_dat(normpts/2+datapts+1:end)];

if VERSION == 1
    % Fit norm line for magnitude
    p_logm = [norm_freq,ones(size(norm_freq))]\norm_logm;
    S21_logm_norm = polyval(p_logm,data_freq);
    % Fit norm line for phase
    p_phase = [norm_freq,ones(size(norm_freq))]\norm_phase;
    S21_phase_norm = polyval(p_phase,data_freq);
    % Complex normalization line
    normline = 10.^(S21_logm_norm./20).*(cos(S21_phase_norm)+1i.*sin(S21_phase_norm));
    
    % Normalize S21
    S21 = 10.^(S21_logm./20).*(cos(S21_phase)+1i.*sin(S21_phase));
    S21_tilde = S21./normline;
    
elseif VERSION == 2
    %Note: Currently doing linear fit, trying to modify to work by
    %normalizing phase in complex directly (would fix unwrapping issues
    %with noise)
    %Normalizing the S21 data for a high Q resonator
    %The normalization is done in the complex plane using off resonance data
    %points.
    %Noise may cause problems.
    %Steps in frequency for should be small enough that change in <S21 per point is
    %less than 180 degrees . Otherwise will get inaccurate results.
    
    %Determines the m*x + b for magnitude and phase
    NormMagInter = [norm_freq norm_freq.^0] \ norm_logm;
    NormPhaseInter = [norm_freq norm_freq.^0] \ norm_phase;
    
    %Uses the left most value from normalization as a zero point, then the
    %slope (m) to remove the linear component due to the transmission line
    %from the magnitude and phase (S21 - zeroPoints - m*(f - f(0)))
    
    S21_logm_norm = norm_logm(1) + NormMagInter(1)*(data_freq-norm_freq(1));
    S21_phase_norm = norm_phase(1) + NormPhaseInter(1)*(data_freq-norm_freq(1));
    
    SParsMagNormed = S21_logm - S21_logm_norm;
    SParsPhaseNormed = S21_phase - S21_phase_norm;
    
    S21_tilde = 10.^(SParsMagNormed/20) .* (exp(1j*SParsPhaseNormed));
end
end
