function [ params, errors ] = fitResonatorInvS21( freq, S21_tilde, guessparams )
%RESONATORFIT Fit the normalized lorentzian of a resonator. Fit
%individually to the real and imaginary parts of S21
% params(1) = Qi
% params(2) = Qc*
% params(3) = phi
% params(4) = f_0

% Calculate S21_tilde_inv
S21_tilde_inv = [real(1./S21_tilde),imag(1./S21_tilde)];

% Initial parameter values
if nargin == 3 % Check if a params variable was passed
    params0 = guessparams;
else % Not arbitrary initial guess
    [~,i] = min(20*log10(abs(S21_tilde)));
    params0 = [1E6,3E5,0.0,freq(i)];
end

% Center and scale manually
params0(1) = params0(1)/1E6; % in millions (10^6)
params0(2) = params0(2)/1E6; % in millions (10^6)
params0(4) = params0(4)/1E9; % in GHz (10^9)

% Fit
warning('off','MATLAB:rankDeficientMatrix');
if false; %exist('nlinfit','file') % use nlinfit if we have the stats toolbox
    [params,~,~,CM,~,~] = ...
        nlinfit(freq,S21_tilde_inv(:),@localResFitScaled,params0);
else
    opts = optimoptions('lsqcurvefit','Display','off',...
        'Algorithm','levenberg-marquardt','ScaleProblem','Jacobian');
    [params,resnorm,~,~,~,~,J] = ...
        lsqcurvefit(@localResFitScaled,params0,freq,S21_tilde_inv(:),[],[],opts);
    % Estimate covariance matrix
    MSE = resnorm/(2*(length(freq)-length(params)));
    CM = full(inv(J'*J))*MSE;
end
warning('on','MATLAB:rankDeficientMatrix');

% Calculate standard errors from covariance matrix
errors = diag(sqrt(CM)).';

% Put parameters and errors back in non-scaled units
params(1) = params(1)*1E6; % in 10^0
params(2) = params(2)*1E6; % in 10^0
params(4) = params(4)*1E9; % in Hz (10^0)
errors(1) = errors(1)*1E6; % in 10^0
errors(2) = errors(2)*1E6; % in 10^0
errors(4) = errors(4)*1E9; % in Hz (10^0)

% Check if the parameters found make sense
if params(1)<0 || params(1)>50E6 || errors(1)>params(1) || params(2)<0 || params(2)>50E6
    error('Fit failed, obtained nonsensical quality factor.');
end
end

function f = localResFitScaled(params,freq)
% This function accepts scaled parameters
f = zeros(length(freq),2);
f(:,1) = 1 + params(1)/params(2) * ( cos(params(3)) + 2*params(1)*(freq-params(4)*1E9)/(params(4)*1E3)*sin(params(3)) )./( 1+( 2*params(1)*(freq-params(4)*1E9)/(params(4)*1E3) ).^2 );
f(:,2) = 0 + params(1)/params(2) * ( sin(params(3)) - 2*params(1)*(freq-params(4)*1E9)/(params(4)*1E3)*cos(params(3)) )./( 1+( 2*params(1)*(freq-params(4)*1E9)/(params(4)*1E3) ).^2 );
f = f(:);
end
