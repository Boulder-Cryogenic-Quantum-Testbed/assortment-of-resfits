function [params,errors,freq,S21data,scans] = fitResonatorInvS21_convertData(VNA_num,figyn,index,tau0)
% Converts data to appropriate form before performing inverse S21 fit.
%% Import data from 'VNA_x.csv' and 'ScanValues.csv'.

scans = dlmread('ScanValues.csv')';
%Tmeas = dlmread('MeasuredTemps.csv');

if exist('PNA_0.csv','file')==0
    name = ['VNA_',int2str(VNA_num),'.csv'];
else
    name = ['PNA_',int2str(VNA_num),'.csv'];
end
temp = dlmread(name);

freq = temp(:,1);
S21mag(1,:) = temp(:,2);
S21ph(1,:) = temp(:,3);

%% Fit to Lorentzian

freq = freq';

S21data = 10.^(S21mag/20).*exp(1i.*S21ph.*pi/180);
S21data = S21data.*exp(2*pi*1i*freq*tau0);

if isempty(index)
    first = ones(length(scans),1);
    last = ones(length(scans),1)*length(S21data(1,:));
    index = [first,last];
end
    
S21data = S21data';
S21mag = S21mag';
S21ph = S21ph';
freq = (10^9)*freq';

%% Fit

[freq,S21_tilde,S21_logm,S21_logm_norm,S21_phase,S21_phase_norm] = resonatorNormalize(freq,S21mag,S21ph,20);

first = index(VNA_num+1,1);
last = index(VNA_num+1,2);
freqclipped = freq(first:last);
S21_tildeclipped = S21_tilde(first:last);

[ params, errors ] = fitResonatorInvS21( freqclipped, S21_tildeclipped );

%% Calculate extra stuff for plots
S21_tilde_inv = [real(1./S21_tilde),imag(1./S21_tilde)];
S21_tilde_inv_fit = resonatorS21TildeInvFun(params,freq);
S21_tilde_fit = 1./(S21_tilde_inv_fit(:,1)+1i*S21_tilde_inv_fit(:,2));
fprime = (freq-params(4))/1E3;

% Plot
opt = {[0.1,0.06]};
subplot = @(m,n,p) subtightplot(m,n,p,opt{:}); 
pos = [360 140 1128 726]; % Should get a 1080x720 fig in the end
f1 = figure('Color','w','Units','pixels','Position',pos);

% Normalization figures
ax1 = subplot(2,3,1);
plot(ax1,freq/1E9,S21_logm,freq/1E9,S21_logm_norm)
ax1.XLim = [freq(1)/1E9 freq(end)/1E9];
ax1.XLabel.String = '$f$ (GHz)';
ax1.YLabel.String = 'Magnitude (dB)';

ax2 = subplot(2,3,4);
plot(ax2,freq/1E9,S21_phase,freq/1E9,S21_phase_norm)
ax2.XLim = [freq(1)/1E9 freq(end)/1E9];
ax2.XLabel.String = '$f$ (GHz)';
ax2.YLabel.String = 'Phase (rad)';
%ax2.YAxis.TickLabelFormat = '%,.1f';

% Fit figures
ax3 = subplot(2,3,2);
plot(ax3,fprime,20*log10(abs(S21_tilde)),fprime,20*log10(abs(S21_tilde_fit)))
ax3.XLim = [min(fprime) max(fprime)];
ax3.XLabel.String = '$f-f_0$ (kHz)';
ax3.YLabel.String = 'Magnitude (dB)';

ax4 = subplot(2,3,5);
plot(ax4,fprime,angle(S21_tilde),fprime,angle(S21_tilde_fit))
ax4.XLim = [min(fprime) max(fprime)];
ax4.XLabel.String = '$f-f_0$ (kHz)';
ax4.YLabel.String = 'Phase (rad)';
%ax4.YAxis.TickLabelFormat = '%,.1f';

ax5 = subplot(2,3,3);
plot(ax5,S21_tilde_inv(:,1),S21_tilde_inv(:,2),S21_tilde_inv_fit(:,1),S21_tilde_inv_fit(:,2))
axis equal
ax5.XLim = [0 ax5.XLim(2)];
ax5.XLabel.String = '$Re[S_{21}^{-1}]$';
ax5.YLabel.String = '$Im[S_{21}^{-1}]$';

% Table with fit parameters (plot invisible data to get axes box)
ax6 = subplot(2,3,6);
plot(ax6,S21_tilde_inv(:,1),S21_tilde_inv(:,2),S21_tilde_inv_fit(:,1),S21_tilde_inv_fit(:,2))
axis equal
ax6.XLim = [0 ax5.XLim(2)];
ax6.XTick = ''; ax6.YTick = '';
ax6.XLabel.String = ''; ax6.YLabel.String = '';
ax6.Children(1).LineStyle = 'none'; ax6.Children(2).LineStyle = 'none';
nums = [params([4,3,1,2]); errors([4,3,1,2])];
nums = nums(:);
str = {'Resonator Fit Results:',''};
str = [str,{sprintf(['\\begin{tabular}{lrr} '...
    'Fit Parameter & Value & Error  \\\\ \\hline '...
    '$f_0$ (Hz)    & %.f   & %.f    \\\\ '...
    '$\\phi$ (rad) & %.3f  & %.3f    \\\\ '...
    '$Q_i$         & %.f   & %.f    \\\\ '...
    '$Q_c^*$       & %.f   & %.f    \\\\ \\hline '...
    '\\end{tabular}\n'],nums),''}];

t = annotation('textbox','String',str,'Interpreter','latex','Margin',10);
t.Position = [0.718 0.175 0.26 0.3];
t.LineStyle = 'none';

% Make fonts bigger
set(findall(f1,'-property','FontSize'),'FontSize',12);
set(findall(f1,'-property','Interpreter'),'Interpreter','latex');

tightfig(f1);

if figyn == 1
    pause
end


end