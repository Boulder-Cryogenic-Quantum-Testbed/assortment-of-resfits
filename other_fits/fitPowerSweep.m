function [ params, errors ] = fitPowerSweep( photons, Qi, T, f_0, params0 )
% Fits the internal quality factor (Qi) of a resonator with resonance 
% frequency (f_0) v.s. average number of photons in the resonator (photons)
% at some constant temperature (T) to the TLS model (TLSPowerDependence).

% Initial fit parameter guesses are set using params0. Good inital guesses:
% params0 = [1E4,1E-3,1,1]; % [Q_hp, Falpha, n_c, beta]

% Returns fit parameters (params), where:
% params(1) = Q_hp
% params(2) = F_alpha
% params(3) = n_c
% params(4) = beta

h  = 6.626069934E-34;
kb = 1.38064852E-23;
tanh_hf0kbT = tanh(h*f_0/(2*kb*T));

% Fit
warning('off','MATLAB:rankDeficientMatrix');
opts = optimoptions('lsqcurvefit','Display','off',...
    'Algorithm','levenberg-marquardt','ScaleProblem','Jacobian');
[params,resnorm,~,~,~,~,J] = ...
    lsqcurvefit(@(params,x)TLSPowerDependence(params,x,tanh_hf0kbT),params0,photons,1./Qi,[],[],opts);
% Estimate covariance matrix
MSE = resnorm/(2*(length(photons)-length(params)));
CM = full(inv(J'*J))*MSE;
warning('on','MATLAB:rankDeficientMatrix');

% Calculate standard errors from covariance matrix
errors = diag(sqrt(CM)).';

end

function f = TLSPowerDependence(params,photons,tanh_hf0kbT)
% Standard TLS power dependence
f = ((params(2)*tanh_hf0kbT)./(1 + photons./params(3)).^params(4) + 1/params(1));
end
