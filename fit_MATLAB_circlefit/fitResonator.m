function [ModelFit,ci,ModelInit,S21bg,freq,S21data,scans] = fitResonator(allscan,VNA_num,figyn,index,tau0)
% fitResonator is used to:
% - import resonator data from VNA
% - fit a hanger geometry resonator to its Q and f0 using the circle fit method,
% - determine the goodness of fit by examining generated plots, and 
% - modify the window of data used to increase the goodness of fit

% Input parameters:
% tau0 = 0;
% 
% allscan = 1;
% VNA_num = 0;
% figyn = 1;
% Also choose index! (below)

%% Import data from 'VNA_x.csv' and 'ScanValues.csv'.

scans = dlmread('ScanValues.csv')';
%Tmeas = dlmread('MeasuredTemps.csv');

if exist('PNA_0.csv','file')==0
    temp = dlmread('VNA_0.csv');
else 
    temp = dlmread('PNA_0.csv');
end
freq = temp(:,1);

S21mag = zeros(length(scans),length(freq));
S21ph = zeros(length(scans),length(freq));

for n = 1:length(scans)
    if exist('PNA_0.csv','file')==0
        name = ['VNA_',int2str(n-1),'.csv'];
    else
        name = ['PNA_',int2str(n-1),'.csv'];
    end
    temp = dlmread(name);
    
    S21mag(n,:) = temp(:,2);
    S21ph(n,:) = temp(:,3);
end

%% Fit to Lorentzian

freq = freq';

S21data = 10.^(S21mag/20).*exp(1i.*S21ph.*pi/180);
S21data = S21data.*exp(2*pi*1i*freq*tau0);

if isempty(index)
    first = ones(length(scans),1);
    last = ones(length(scans),1)*length(S21data(1,:));
    index = [first,last];
end
    
if allscan == 1
    for k = 1:length(scans)
        first = index(k,1); 
        last = index(k,2);
        [ModelFit{k},ci{k},ModelInit{k},S21bg{k}] = hangerS21fit(freq(first:last),S21data(k,first:last),[],[],[],figyn);
        if figyn == 1
            pause
        end
    end
elseif allscan == 0
    first = index(1,1);
    last = index(1,2);
    [ModelFit{VNA_num+1},ci{VNA_num+1},ModelInit{VNA_num+1},S21bg{VNA_num+1}] = hangerS21fit(freq(first:last),S21data(VNA_num+1,first:last),[],[],[],figyn);
else
    print('Error: allscan must be 0 or 1.')
end
end