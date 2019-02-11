# -*- coding: utf-8 -*-
"""
Created on Mon May 14 19:32:43 2018

@author: hung93
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands
import sys #update paths
sys.path.append(r'\\kdoserver\Data\Software\Python\Fit_Cavity_ShareWithPappas')#path to Fit_Cavity
from Fit_Cavity import *
np.set_printoptions(precision=4,suppress=True)# display numbers with 4 sig. figures (digits)          

                         ## Code Start Here ##
                    
plt.close("all")
dic = r'\\kdoserver\Data\Projects\LY181223_ly_aSi\ly190202_LeidenRun\freq7.151G\RAP_-65dB_Vpp0.5V_to200k_sym1_span4M'
filename = '20190206_053441_Pow-25_Bias501.19.txt'

filepath = dic+'\\'+filename

## Single file
delay = 100.77 #ns
data = np.genfromtxt(filepath)
xdata = data.T[0]/10**9   ## make sure frequency is in GHz##
y1data = data.T[1]          ## I
y2data = data.T[2]          ## Q
ydata= y1data +1j*y2data
ydata = np.multiply(ydata,np.exp(1j*delay*2*np.pi*xdata))
y1data1 = np.real(ydata)
y2data1 = np.imag(ydata)
ResA = resonator(xdata, y1data, y2data, 0, delay = delay) # create resonator class
DCM_result,INV_result,_  = ResA.fit(extract_factor = 4,DCM_init_guess = None,INV_init_guess = None)

plt.close("all")
DCM_result[1].savefig(r'Z:\Software\Python\Fit_Cavity_ShareWithPappas\ExampleData\DCM_example fig.png')
INV_result[1].savefig(r'Z:\Software\Python\Fit_Cavity_ShareWithPappas\ExampleData\INV_example fig.png')
