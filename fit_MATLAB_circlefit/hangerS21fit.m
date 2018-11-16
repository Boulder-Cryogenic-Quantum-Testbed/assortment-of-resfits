function [ModelFit,ci,ModelInit,S21bg] = hangerS21fit(freq,S21data,c0,lb,ub,fig)
% hangerS21fit performs a fitting of S21 parameters for a hanger geometry
% resonator.

%% Calculating atanfit init if no input
if isempty(c0) && isempty(lb) && isempty(ub)
    [f0,df,Q0] = atanfit_init(freq,S21data);
    c0 = [pi,Q0,f0]; % initial guess
    lb = [0,1,f0-3*df]; % lower bound
    ub = [2*pi,1000,f0+3*df]; % upper bound
end

%% Assume electrical delay Model.tau has already been removed by VNA
ModelInit.tau = 0;

%% Find the off resonace reference point and normalize the raw data
% Estimate Model.a and Model.alpha
[fitr1,fitc1] = cirfit(S21data);
S21temp = S21data - fitc1;
x = freq;
y = unwrap(angle(S21temp));

atanfit1 = atanfit(x,y,c0,lb,ub);

S21bg = fitr1*exp(1i*(atanfit1(atanfit1.fr)-pi)) + fitc1;
S21nor = S21data/S21bg;

ModelInit.a = abs(S21bg);
ModelInit.alpha = angle(S21bg);

%% Performing circle fit on normalized S21 data
% In this step, I estimate Model.Qr/Model.Qc = 2r
[fitr2,fitc2] = cirfit(S21nor);
%%  Performing atan fit on shiftted and rotated circle 
% In this step, I estimate Model.fr, Model.Qr, Model.Qc, and Model.phi
S21trans = (fitc2 - S21nor)*exp(-1i*angle(fitc2));

x = freq;
y = unwrap(angle(S21trans));
atanfit2 = atanfit(x,y,c0,lb,ub);

ModelInit.fr = atanfit2.fr;
ModelInit.Qr = atanfit2.Qr;
ModelInit.Qc = ModelInit.Qr/2/fitr2;
ModelInit.phi = angle(exp(1i*(-atanfit2.theta0 + angle(fitc2))));

%% Hanger S21 complx function fit
[ModelFit,ci] = hangerS21cplxfit(freq,S21data,ModelInit,[],[]);
S21Fit = hangerS21(freq',ModelFit);
Qi = (1/ModelFit.Qr-1/ModelFit.Qc)^-1;

%% figure
if fig == 1
    hf10 = figure(10);
    hf10.Position = [50 350 1000 600];
    clf
    subplot(2,3,[1,4],'position',[0.11,0.2400,0.22,0.82])
    hold on
    plot(real(S21Fit),imag(S21Fit),'r','linewidth',2)
    plot(real(S21data),imag(S21data),'b.','markersize',6)
    hold off
    grid on
%     axis([-1.2,1.2,-1.2,1.2])
    axis square
    title('Real/Imag')
    xlabel('Real(S21)')
    ylabel('Imag(S21)')
    subplot(2,3,2)
    hold on
    plot(freq,20*log10(abs(S21Fit)),'r','linewidth',2)
    plot(freq,20*log10(abs(S21data)),'b.','markersize',6)
    hold off
    grid on
    title('S21 magnitude')
    xlabel('Frequency (GHz)')
    ylabel('Magnitude (dB)')
    subplot(2,3,3)
    hold on
    plot(freq,unwrap(angle(S21Fit))*180/pi,'r','linewidth',2)
    plot(freq,unwrap(angle(S21data))*180/pi,'b.','markersize',6)
    hold off
    grid on
    title('S21 phase')
    xlabel('Frequency (GHz)')
    ylabel('Phase (degree)')
    subplot(2,3,5)
    hold on
    plot(freq,20*log10(abs(S21data))-20*log10(abs(S21Fit)),'b','linewidth',2)
    hold off
    grid on
    title('S21 magnitude fit residuals')
    xlabel('Frequency (GHz)')
    ylabel('Magnitude (dB)')
    subplot(2,3,6)
    hold on
    plot(freq,(unwrap(angle(S21data))-unwrap(angle(S21Fit)))*180/pi,'b','linewidth',2)
    hold off
    grid on
    title('S21 phase fit residuals')
    xlabel('Frequency (GHz)')
    ylabel('Phase (degree)')
    tabledata = {'fr',ModelFit.fr,'  GHz';'Qr',ModelFit.Qr,'  k';'Qc',ModelFit.Qc,'  k';'Qi',Qi,'  k'};
    uitable(gcf,'data',tabledata,'ColumnName',{'Parameter','Value','Unit'},'FontSize',10,'Position',[60 80 280 120]);
end

%% Local function definitions

function [f0,df,Q0] = atanfit_init(freq,S21data)
% Innitial fitting
% Chooses the minimum of the S21 magnitude as the resonant frequency f0.
% Determines the resonator bandwidth by unwrapping the phase angle.
% Estimates the Q using these values.
[~,loc0] = min(20*log10(abs(S21data)));
[~,loc1] = min(unwrap(angle(S21data)));
[~,loc2] = max(unwrap(angle(S21data)));
f0 = freq(loc0);
df = abs(freq(loc2) - freq(loc1));
Q0 = f0/df;
end

function atanfitout = atanfit(x,y,c0,lb,ub)
%UNTITLED5 Summary of this function goes here
%   x: freq (GHz), y: unwrap(angle(S21)) (radian) (center at origin);
%   coefficients: theta0 (radian), Qr (k), fr (GHz)
fo = fitoptions('Method','NonlinearLeastSquares','StartPoint',c0,'Lower',lb,'Upper',ub);
ft = fittype('-theta0+2*atan(2*Qr*1e3*(1-x/fr))','coefficient',{'theta0','Qr','fr'},'options',fo);
atanfitout = fit(x',y',ft);
end

function [fitr,fitc] = cirfit(data)
% Circle fit
%   Detailed explanation goes here
M = zeros(4,4);
x = real(data);
y = imag(data);
z = x.^2 + y.^2;
n = length(data);
M(1,1) = sum(z.*z);
M(1,2) = sum(x.*z);
M(2,2) = sum(x.*x);
M(1,3) = sum(y.*z);
M(2,3) = sum(x.*y);
M(3,3) = sum(y.*y);
M(1,4) = sum(z);
M(2,4) = sum(x);
M(3,4) = sum(y);
M(4,4) = n;
M(2,1) = M(1,2);
M(3,1) = M(1,3);
M(4,1) = M(1,4);
M(3,2) = M(2,3);
M(4,2) = M(2,4);
M(4,3) = M(3,4);
B = [0 0 0 -2;0 1 0 0;0 0 1 0;-2 0 0 0];
[V,D] = eig(M,B);
D(D<=0) = Inf;
[~,I] = min(min(D));
A = V(:,I);
fitr = 1/2/abs(A(1))*sqrt(A(2)^2+A(3)^2-4*A(1)*A(4));
fitc = -A(2)/2/A(1) - 1i*A(3)/2/A(1);
end

function [ModelFit,ci] = hangerS21cplxfit(freq,S21data,ModelInit,lb,ub)
%UNTITLED7 Summary of this function goes here
%   freq (GHz): scanned freucney;
%   ModelInit.tau (ns): electrical delay of the input line;
%   ModelInit.a (~): attenuation of the input line;
%   ModelInit.alpha (radian): initial phase of the input line;
%   ModelInit.fr (GHz): resonance frequency;
%   ModelInit.Qr (k): total Q of the resonator;
%   ModelInit.Qc (k): coupling Q of the resonator;
%   ModelInit.phi (radian): impedence mismatch;
xdata = freq';
ydata2 = [real(S21data'),imag(S21data)'];
V0 = [ModelInit.tau;ModelInit.a;ModelInit.alpha;ModelInit.fr;ModelInit.Qr;ModelInit.Qc;ModelInit.phi];
opts = optimoptions(@lsqcurvefit,'Display','off');
% [Vfit,resnorm,residual,exitflag,output,lambda,jacobian] = lsqcurvefit(@obj_hangerS21,V0,xdata,ydata2,lb,ub,opts);
[Vfit,~,residual,~,~,~,jacobian] = lsqcurvefit(@obj_hangerS21,V0,xdata,ydata2,lb,ub,opts);
ci = nlparci(Vfit,residual,'jacobian',jacobian);
ModelFit.tau = Vfit(1);
ModelFit.a = Vfit(2);
ModelFit.alpha = Vfit(3);
ModelFit.fr = Vfit(4);
ModelFit.Qr = Vfit(5);
ModelFit.Qc = Vfit(6);
ModelFit.phi = Vfit(7);
end

function yout = obj_hangerS21(V,x)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
freq = x;
Model.tau = V(1);
Model.a = V(2);
Model.alpha = V(3);
Model.fr = V(4);
Model.Qr = V(5);
Model.Qc = V(6);
Model.phi = V(7);
hangerS21 = Model.a*exp(1i*Model.alpha)*exp(-2*pi*1i*freq*Model.tau).*(1-((Model.Qr/Model.Qc)*exp(1i*Model.phi))./(1+2i*Model.Qr*1e3*(freq/Model.fr-1)));

yout = zeros(length(x),2);
yout(:,1) = real(hangerS21);
yout(:,2) = imag(hangerS21);
end

function hangerS21 = hangerS21(freq,Model)
% Calculate S21 of hanger geometry
%   freq (GHz): scanned frequency;
%   Model.tau (ns): electrical delay of the input line;
%   Model.a (~): attenuation of the input line;
%   Model.alpha (radians): initial phase of the input line;
%   Model.fr (GHz): resonance frequency;
%   Model.Qr (k): total Q of the resonator;
%   Model.Qc (k): coupling Q of the resonator;
%   Model.phi (radians): impedence mismatch;

% Generating S21 of hanger geometry
hangerS21 = Model.a*exp(1i*Model.alpha)*exp(-2*pi*1i*freq*Model.tau).*(1-((Model.Qr/Model.Qc)*exp(1i*Model.phi))./(1+2i*Model.Qr*1e3*(freq/Model.fr-1)));
end

end

