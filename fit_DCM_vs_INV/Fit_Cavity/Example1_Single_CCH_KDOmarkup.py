# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:32:43 2018

@author: hung93
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands
import sys #update paths
sys.path.append(r'Z:\Software\Python\Fit_Cavity_ShareWithPappas')#path to Fit_Cavity
from Fit_Cavity import *
np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig. figures (digits)          

                         ## Code Start Here ##
                    
plt.close("all")
dic = r'Z:\Software\Python\Fit_Cavity_ShareWithPappas\ExampleData' #folder with data
filename = 'Data_delay 99p6 ns.txt' #datafile in above folder

filepath = dic+'\\'+filename 

#############################################
## create Method

delay = 99.6 # ns , electrical delay 
extract_factor = 1.5 
MC_rounds = 1e3
frange = [0,10]
find_circle = True 
manual_init = [0.05,
 5000,
 1000,
 4.724838317959076,
 0.04513820567542649,
 5]#make your own initla guess: [amplitude, Q, Qe, freq, phi, theta]
manual_init = None # find initial guess by itself
power_calibration = [0,0,0]
MC_fix = ['w1']
Method = Fit_Method('INV',extract_factor,1,MC_rounds = MC_rounds,\
                 MC_fix = MC_fix,find_circle = find_circle,\
                 manual_init=manual_init,MC_step_const= 0.3)
#Method.MC_weight = 'no'
#Method.MC_rounds = 10
#Method.MC_iteration = 1e6
##############################################################
## Single file
data = np.genfromtxt(filepath)
xdata = data.T[0]/10**9   ## make sure frequency is in GHz##
y1data = data.T[1]          ## I
y2data = data.T[2]          ## Q
ydata= y1data +1j*y2data
ydata = np.multiply(ydata,np.exp(1j*delay*2*np.pi*xdata))
y1data1 = np.real(ydata)
y2data1 = np.imag(ydata)
ResA = resonator(xdata, y1data, y2data, 100, delay = delay) #save measuremene result using resonator class (with "." names); 100 used as fake power (dBm)
params1,fig1,chi1,init1 = Fit_Resonator(ResA,Method) # Fit Resonator A with method, get fit result=params, fig1 is figure, chi is chi^2, init=initial_guess 
#ResA.load_params(Method.method,Params,chi)                 # load param. to Resonator A , useful below if you analyze number of photons
#ResA.power_calibrate(power_calibration)                      # calibrate device power, get information about number of photons 

## change fitting method to inverse S21
#Method.change_method('INV')          
#Method.manual_init = None              # change method to fit inverse S21 
#params2,fig2,chi2,init2 = Fit_Resonator(ResA,Method)
#ResA.load_params(Method.method,Params,chi)
#print('DCM parameters',ResA.DCMparams.__dict__)
#print('INV parameters',ResA.INVparams.__dict__)
fig1.savefig(dic+'\\Fig\\'+filename+'_fit.png')
###############################################
