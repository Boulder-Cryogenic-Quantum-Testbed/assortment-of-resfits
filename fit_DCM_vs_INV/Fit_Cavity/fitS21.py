# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:37:41 2018

@author: hung93
"""
import numpy as np
import lmfit
import emcee
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sympy as sym
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from lmfit import Minimizer
import inflect
import matplotlib.pylab as pylab
from scipy import optimize
import emcee
import time

params = {'legend.fontsize': 18,
          'figure.figsize': (10, 8),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize':18,
         'ytick.labelsize':18,
         'lines.markersize' : 1,
         'lines.linewidth' : 2,
         'font.size': 15.0 }
pylab.rcParams.update(params)

np.set_printoptions(precision=4,suppress=True)
p = inflect.engine() # search ordinal
#####################################################################
## Data related
#####################################################################
def extract_data(x,y,x1,x2):
    x_temp = []
    y_temp = []
    for i in range(len(x)):
        if (x[i] >x1 and x[i]<x2):
            x_temp.append(x[i])
            y_temp.append(y[i])
    return np.asarray(x_temp),np.asarray(y_temp)
#########################################################################
def convert_params(from_method,params):
    if from_method =='DCM':
        Qe = params[2]/np.cos(params[4])
        Qi = params[1]*Qe/(Qe-params[1])
        Qe_INV = params[2]
        Qi_INV = Qi/(1+np.sin(params[4])/Qe_INV/2)
        return [1/params[0],Qi_INV,Qe_INV,params[3],-params[4],-params[5]]
    elif from_method == 'INV':
        Qe_DCM = params[2]
        Q_DCM = ( np.cos(params[4])/params[2]+1/params[1])**-1
        return [1/params[0],Q_DCM,Qe_DCM,params[3],-params[4],-params[5]]

 
########################################################################
def Find_Circle(y1,y2,k = 1):    
    y_m = np.abs(y1 +1j*y2)
    idx = np.argmin(np.log10(np.abs(y1 +1j*y2)))
    y1c = y1[idx]
    y2c = y2[idx]
    y1 = y1-y1[idx]
    y2 = y2-y2[idx]
    y_m = np.abs(y1 +1j*y2)

    #Least Squares Fitting of Circles
    # by N. CHERNOV AND C. LESORT
    w = y1**2+y2**2
    M11 = np.sum(w**2)
    M12 = np.sum(y1*w)
    M13 = np.sum(y2*w)
    M14 = np.sum(w)
    M22 = np.sum(y1**2)
    M23 = np.sum(y1*y2)
    M24 = np.sum(y1)
    M33 = np.sum(y2**2)
    M34 = np.sum(y2)
    M44 = len(w)
    radius_guess = np.max(y_m)
    
    try:
        ## Find smallest positive solution 'x' for det(M-xB)
        xx = sym.Symbol('xx')
        BB = sym.Matrix([[0,0,0,-2],[0,1,0,0],[0,0,1,0],[-2,0,0,0]])
        MM = sym.Matrix([[M11, M12,M13,M14],[M12,M22,M23,M24],[M13,M23,M33,M34],[M14,M24,M34,M44]])
        A = sym.Matrix(MM-xx*BB)
        DA = A.det()
        x_roots = sym.solveset(DA,xx)
        x_roots = list(x_roots)
    except:
        MM = np.array([[M11, M12,M13,M14],[M12,M22,M23,M24],[M13,M23,M33,M34],[M14,M24,M34,M44]])
        BB = np.array([[0,0,0,-2],[0,1,0,0],[0,0,1,0],[-2,0,0,0]])
        f = lambda xx :np.linalg.det(MM-xx*BB)
        x_roots = optimize.fsolve(f,0)
        
    


    x_c_array = []
    y_c_array = []
    radius_array = []
    error = []

    for k in range(len(x_roots)): 
        Temp1 = np.matrix(MM-x_roots[k]*BB)
        Temp2 = np.array(Temp1).astype(np.float64)
        w,v = np.linalg.eig(Temp2)
        right_v = v[:,1] # eigenvector we want
        x_c = -right_v[1]/right_v[0]/2
        y_c = -right_v[2]/right_v[0]/2
        radius = np.average(((y1-x_c)**2+(y2-y_c)**2)**0.5)
        if (x_c**2+y_c**2)**0.5 > np.max(y_m):
            error.append(np.inf)
        else:
            error.append(abs(radius-radius_guess))
        x_c = x_c+y1c
        y_c = y_c+y2c

        x_c_array.append(x_c)
        y_c_array.append(y_c)
        radius_array.append(radius)
    correct_index = error.index(min(error))
    if len(error)!= 1 and error[2] < error[1]*2:
        correct_index = 1
    else:
        pass
#    print(correct_index)
    x_c = x_c_array[correct_index]
    y_c = y_c_array[correct_index]
    radius = radius_array[correct_index]
    return x_c,y_c,radius
#########################################################################

def normalize(y1data,y2data):
    ## Get amp and theta guess
    ydata = np.abs(y1data +1j*y2data)
    idx = np.argmax(ydata)
    Amp = np.sum(ydata[0]+ydata[-1])/2
    theta1 = np.angle(y1data[0] +1j*y2data[0])
    theta2 = np.angle(y1data[-1] +1j*y2data[-1])
    theta = (theta1+theta2)/2
    ydata = y1data + 1j*y2data
    Norm_ydata = ydata/Amp*np.exp(-1j*theta)
    return Norm_ydata,Amp, theta, idx

########################################################################

def renormalize(xdata,init,norm_y):
    ## get kappa, gamma, phi guess
    temp_y = -norm_y + 1
    freq_idx = np.argmax(np.abs(temp_y))
    freq_guess = xdata[freq_idx]
    max_temp_y = np.abs(temp_y[freq_idx])
    phi_guess = np.angle(temp_y[freq_idx])
    init[3] = freq_guess
    _,k_r_idx = find_nearest(np.abs(temp_y),max_temp_y/2)
    kappa_guess = abs((xdata[freq_idx] - xdata[k_r_idx]))*2**0.5*max_temp_y
    gamma_guess = abs((xdata[freq_idx] - xdata[k_r_idx]))*2**0.5-kappa_guess
    #Sometimes gamma gets negative number
    gamma_guess = abs(gamma_guess)
    init[1] = kappa_guess
    init[2] = gamma_guess
    init[4] = phi_guess
    
    return init

#########################################################################
   
def Find_initial_guess(x,y1,y2,Method,k = 1):

    y = y1 +1j*y2
    freq_idx = np.argmin(np.abs(y))
    f_c = x[freq_idx] ###############4

    if Method.method == 'INV':
        y = 1/y
        
    y1 = np.real(y)
    y2 = np.imag(y)
    x_c,y_c,r = Find_Circle(y1,y2,k)
    z_c = x_c+1j*y_c
    
    
    x_s1 = np.average(np.real(y)[0:10])
    x_s2 = np.average(np.real(y)[-10:])
    y_s1 = np.average(np.imag(y)[0:10])
    y_s2 = np.average(np.imag(y)[-10:])
    
    z_1 = (x_s1+x_s2)/2 + 1j*(y_s1+y_s2)/2
    Amp = abs(z_1) #############1
    
    vector = z_1-z_c
    z_1 = z_c + vector/abs(vector)*r
    
    theta = np.angle(z_1)   ############6
    ## move gap of circle to (0,0)
    ydata = y*np.exp(-1j*theta) 
    ydata = ydata/Amp
    ydata = ydata-1
    
    z_c = z_c*np.exp(-1j*theta)
    z_c = z_c/Amp
    z_c = z_c -1
    if Method.method == 'INV':
        phi = np.angle(z_c)  ###########5
        phi = -phi
    else:
        phi = np.angle(-z_c)  ###########5

    # rotate resonant freq to minimum
    ydata = ydata*np.exp(-1j*phi)

    if Method.method == 'DCM':
        
        Q_Qe = np.max(np.abs(ydata))
        y_temp =np.abs(np.abs(ydata)-np.max(np.abs(ydata))/2**0.5)
        
        _,idx1 = find_nearest(y_temp[0:freq_idx],0)
        _,idx2 = find_nearest(y_temp[freq_idx:],0)
        idx2 = idx2+freq_idx
        kappa = abs((x[idx1] - x[idx2]))
        
        Q = f_c/kappa
        Qe = Q/Q_Qe
        popt, pcov = curve_fit(One_Cavity_peak_abs, x,np.abs(ydata),p0 = [Q,Qe,f_c],bounds = (0,[np.inf]*3))
        Q = popt[0]
        Qe = popt[1]
        init_guess = [Amp,Q,Qe,f_c,phi,theta]
        
    elif Method.method == 'INV':
        
        Qi_Qe = np.max(np.abs(ydata))
        
        y_temp =np.abs(np.abs(ydata)-np.max(np.abs(ydata))/2**0.5)
        
        _,idx1 = find_nearest(y_temp[0:freq_idx],0)
        _,idx2 = find_nearest(y_temp[freq_idx:],0)
        idx2 = idx2+freq_idx
        kappa = abs((x[idx1] - x[idx2]))
        
        Qi = f_c/kappa
        Qe = Qi/Qi_Qe
        popt, pcov = curve_fit(One_Cavity_peak_abs, x,np.abs(ydata),p0 = [Qi,Qe,f_c],bounds = (0,[np.inf]*3))
        Qi = popt[0]
        Qe = popt[1]
        init_guess = [Amp,Qi,Qe,f_c,phi,theta]
        
    elif Method.method == 'CPZM':
        
        Q_Qe = np.max(np.abs(ydata))
        y_temp =np.abs(np.abs(ydata)-np.max(np.abs(ydata))/2**0.5)
        
        _,idx1 = find_nearest(y_temp[0:freq_idx],0)
        _,idx2 = find_nearest(y_temp[freq_idx:],0)
        idx2 = idx2+freq_idx
        kappa = abs((x[idx1] - x[idx2]))
        
        Q = f_c/kappa
        Qe = Q/Q_Qe
        popt, pcov = curve_fit(One_Cavity_peak_abs, x,np.abs(ydata),p0 = [Q,Qe,f_c],bounds = (0,[np.inf]*3))
        Q = popt[0]
        Qe = popt[1]
        Qa = -1/np.imag(Qe**-1*np.exp(-1j*phi))
        Qe = 1/np.real(Qe**-1*np.exp(-1j*phi))
        Qi = (1/Q-1/Qe)**-1
        Qie = Qi/Qe
        Qia = Qi/Qa
        init_guess = [Amp,Qi,Qie,f_c,Qia,theta]
    return init_guess,x_c,y_c,r

########################################################################
def Find_ElectricDelay(xdata,ydata):
    angdata = np.unwrap(np.angle(ydata))
    
    ## find 5% data from start and end
    findex_5pc = int(len(xdata)*0.05)
    freqEnds = np.concatenate((xdata[0:findex_5pc], xdata[-findex_5pc:-1]))
    phaseEnds = np.concatenate((angdata[0:findex_5pc], angdata[-findex_5pc:-1]))
    ## fit phase and get
    phaseBaseCoefs = np.polyfit(freqEnds, phaseEnds, 1)
    ED_ydata = np.multiply(ydata,np.exp(1j*-phaseBaseCoefs[0]*xdata))
    ## Electric delay in ns (i*2*pi*ED_time*f)
    ED_time = phaseBaseCoefs[0]/2/np.pi
    print('Electrical delay time = '+str(ED_time)+' (ns)')
    return ED_ydata,ED_time
     
########################################################################
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    val = array[idx]
    return val, idx

###########################################################
   ## PLOT REAL and IMAG figure 
def PlotFit(x,y,params,func,figurename,x_c,y_c,radius,extract_factor = None):
    plt.close(figurename)
    if extract_factor == None:
        x_fit = np.linspace(x.min(),x.max(),5000)        
    elif isinstance(extract_factor, list) == True: 
        x_fit = np.linspace(extract_factor[0],extract_factor[1],5000)
        y_fit = func(x_fit,*params)
    y_fit = func(x_fit,*params)
    y_noise = func(x,*params)
    noise = np.linalg.norm(y_noise-y)
    SNR = noise/len(x)/params[0]            ## think about SNR
    
    fig = plt.figure(figurename,figsize=(15, 10))
    gs = GridSpec(3,3)   
    ax1 = plt.subplot(gs[0,0]) ## real part
    ax2 = plt.subplot(gs[0,1]) ## imag part
    ax3 = plt.subplot(gs[0,2]) ## magnitude
    ax4 = plt.subplot(gs[1,2]) ## angle
    ax = plt.subplot(gs[1:3,0:2]) ## IQ plot 
    
    if isinstance(extract_factor, list) == True:
        x_fit = np.linspace(x.min(),x.max(),5000)        
        y_fit = func(x_fit,*params)
    
        ax1.plot(x_fit,np.real(y_fit),'g--',label = 'final fit')
        ax2.plot(x_fit,np.imag(y_fit),'g--',label = 'final fit')
        ax3.plot(x_fit,np.abs(y_fit),'g--',label = 'final fit')
        ax4.plot(x_fit,np.angle(y_fit),'g--',label = 'final fit')
        ax.plot(np.real(y_fit),np.imag(y_fit),'--',color = 'lightgreen',label = 'guide line',linewidth = 4.5)

        x_fit = np.linspace(extract_factor[0],extract_factor[1],5000)
        y_fit = func(x_fit,*params)

        
    ax1.plot(x,np.real(y),'bo',label = 'raw')
    ax1.plot(x_fit,np.real(y_fit),'g-',label = 'final fit')
    ax1.set_xlim(left=x[0], right=x[-1])
    ax1.set_title("real part S21")
    if func == Cavity_inverse:
        ax1.set_title("real part $S_{21}^{-1}$")
    plt.ylabel('Re[S21]')
    plt.xlabel('frequency (GHz)')
#    ax1.legend()
    
    ax2.plot(x,np.imag(y),'bo',label = 'raw')
    ax2.plot(x_fit,np.imag(y_fit),'g-',label = 'final fit')
    ax2.set_xlim(left=x[0], right=x[-1])
    ax2.set_title("Imag part S21")
    if func == Cavity_inverse:
        ax2.set_title("Imag part $S_{21}^{-1}$")
    plt.ylabel('Im[S21]')
    plt.xlabel('frequency (GHz)')
#    ax2.legend()

    ax3.plot(x,np.abs(y),'bo',label = 'raw')
    ax3.plot(x_fit,np.abs(y_fit),'g-',label = 'final fit')
    ax3.set_xlim(left=x[0], right=x[-1])
    ax3.set_title("Magnitude part S21")
    if func == Cavity_inverse:
        ax3.set_title("Magnitude part $S_{21}^{-1}$")
    plt.ylabel('Abs[S21]')
    plt.xlabel('frequency (GHz)')
#    ax3.legend()
    
    ax4.plot(x,np.angle(y),'bo',label = 'raw')
    ax4.plot(x_fit,np.angle(y_fit),'g-',label = 'final fit')
    ax4.set_xlim(left=x[0], right=x[-1])
    ax4.set_title("Angle part S21")
    if func == Cavity_inverse:
        ax4.set_title("Angle part $S_{21}^{-1}$")
    plt.ylabel('Ang[S21]')
    plt.xlabel('frequency (GHz)')
    


    line1 = ax.plot(np.real(y),np.imag(y),'bo',label = 'raw data',markersize = 3)
    line2 = ax.plot(np.real(y_fit),np.imag(y_fit),'g-',label = 'final fit',linewidth = 6)
    if x_c ==0 and y_c ==0 and radius == 0:
        pass
    
    else:
        plt.plot(x_c,y_c,'*',markersize = 5,color=(0, 0.8, 0.8))
        circle = Circle((x_c, y_c), radius, facecolor='none',\
                    edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    
    plt.axis('square')
    plt.title("Real and Imag part S21")
    if func == Cavity_inverse:
        plt.title("Real and Imag part $S_{21}^{-1}$")
    plt.ylabel('Im[S21]')
    plt.xlabel("Re[S21]")
    if func == Cavity_inverse:
         plt.ylabel('Im[$S_{21}^{-1}$]')
         plt.xlabel("Re[$S_{21}^{-1}$]")  
    leg = plt.legend()
    
# get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(10)
        
    if params != []:
        if func == Cavity_inverse:
            
            textstr = r'$Q_e$ = '+'{0:05f}'.format(params[2]) + \
            '\n' + r'$Q_i$ = '+'{0:05f}'.format(params[1])+\
            '\n' + r'$f_c$ = '+'{0:05f}'.format(params[3])+\
            '\n' + r'$\phi$ = '+'{0:05f}'.format(params[4]/np.pi*180)+' $\degree$'+\
            '\n' + 'SNR = ' + '{0:05f}'.format(SNR)
        elif func == Cavity_CPZM:
            textstr = r'$Q_e$ = '+'{0:05f}'.format(params[1]*params[2]**-1) + \
            '\n' + r'$Q_i$ = '+'{0:05f}'.format(params[1])+\
            '\n' + r'$Q_a$ = '+'{0:05f}'.format(params[1]*params[4]**-1)+\
            '\n' + r'$f_c$ = '+'{0:05f}'.format(params[3])+\
            '\n' + 'SNR = ' + '{0:05f}'.format(SNR)
        else:
            Qe = params[2]*np.exp(1j*params[4])
            Qi = (params[1]**-1-abs(np.real(Qe**-1)))**-1
            textstr = 'Q = '+ '{0:05f}'.format(params[1]) + \
            '\n' + r'1/Re[1/$Q_e$] = ' +'{0:05f}'.format(1/np.real(1/Qe)) + \
            '\n' + r'$Q_e$ = '+'{0:05f}'.format(params[2]) + \
            '\n' + r'$Q_i$ = '+'{0:05f}'.format(Qi)+\
            '\n' + r'$f_c$ = '+'{0:05f}'.format(params[3])+\
            '\n' + r'$\phi$ = '+'{0:05f}'.format(params[4]/np.pi*180)+' $\degree$'+\
            '\n' + 'SNR = ' + '{0:05f}'.format(SNR)
        
        plt.gcf().text(0.7, 0.1, textstr, fontsize=18)
   
    plt.tight_layout()
    return fig

#######################################################################
## Fit Function
#########################################

def Cavity_DCM(x, A, Q, Qe, w1,phi,theta):
    return np.array(A*np.exp(1j*theta)*(1-Q/Qe*np.exp(1j*phi)/(1 + 1j*(x-w1)/w1*2*Q)))

def Cavity_inverse(x, A, Qi,Qe, w1,phi,theta):
    return np.array(\
        A*np.exp(1j*theta)*(1 + Qi/Qe*np.exp(-1j*phi)/(1 + 1j*2*Qi*(x-w1)/w1)))   
def Cavity_CPZM(x, A, Qi,Qie,w1,Qia,theta):
    return np.array(\
        A*np.exp(1j*theta)*(1 + 2*1j*Qi*(x-w1)/w1)/(1 + Qie +1j*Qia + 1j*2*Qi*(x-w1)/w1))   

def One_Cavity_peak_abs(x, Q, Qe, w1):
    return np.abs(Q/Qe/(1 + 1j*(x-w1)/w1*2*Q))

#############################################################################
def fit_raw_compare(x,y,params,method):
    if method == 'DCM':
        func = Cavity_DCM
    if method == 'INV':
        func = Cavity_inverse
    yfit = func(x,*params)
    ym = np.abs(y-yfit)/np.abs(y)
    return ym
############################################################################
    ## Least square fit model for one cavity, dip-like S21
def min_one_Cavity_dip(parameter, x, data=None):
    A = parameter['amp']
    Q = parameter['Q']
    Qe = parameter['Qe']
    w1 = parameter['w1']
    phi = parameter['phi']
    theta = parameter['theta']
    
    model = Cavity_DCM(x, A, Q, Qe, w1,phi,theta)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag
    
    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))

########################################################
    ## Least square fit model for one cavity, dip-like "Inverse" S21
def min_one_Cavity_inverse(parameter, x, data=None):
    A = parameter['amp']
    Qe = parameter['Qe']
    Qi = parameter['Qi']
    w1 = parameter['w1']
    phi = parameter['phi']
    theta = parameter['theta']
    
    model = Cavity_inverse(x, A, Qi, Qe, w1,phi,theta)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag
    
    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))


#####################################################################
def min_one_Cavity_CPZM(parameter, x, data=None):
    A = parameter['amp']
    Qe = parameter['Qe']
    Qi = parameter['Qi']
    w1 = parameter['w1']
    Qa = parameter['Qa']
    theta = parameter['theta']
    
    model = Cavity_inverse(x, A, Qi, Qe, w1,Qa,theta)
    real_model = model.real
    imag_model = model.imag
    real_data = data.real
    imag_data = data.imag
    
    resid_re = real_model - real_data
    resid_im = imag_model - imag_data
    return np.concatenate((resid_re,resid_im))


#####################################################################
def MonteCarloFit(xdata= None,ydata=None,parameter=None,Method = None):
    
    
    ydata_1stfit = Method.func(xdata, *parameter)
        
    ## weight condition
    if Method.MC_weight == 'yes':
        weight_array = 1/abs(ydata)
    else:
        weight_array = np.full(len(xdata),1)
    
    weighted_ydata = np.multiply(weight_array,ydata)
    weighted_ydata_1stfit = np.multiply(weight_array,ydata_1stfit)
    error = np.linalg.norm(weighted_ydata - weighted_ydata_1stfit)/len(xdata) # first error 
    error_0 = error
    ## Fix condition and Monte Carlo Method with random number Generator
    
    counts = 0
    while counts < Method.MC_rounds:
        counts = counts +1
        random = Method.MC_step_const*(np.random.random_sample(len(parameter))-0.5)
       ## Fix parameter on demand
        if 'Amp' in Method.MC_fix:
            random[0] = 0
        if 'Q' in Method.MC_fix:
            random[1] = 0
        if 'Qi' in Method.MC_fix:
            random[1] = 0
        if 'Qe' in Method.MC_fix:
            random[2] = 0
        if 'w1' in Method.MC_fix:
            random[3] = 0
        if 'phi' in Method.MC_fix:
            random[4] = 0
        if 'Qa' in Method.MC_fix:
            random[4] = 0
        if 'theta' in Method.MC_fix:
            random[5] = 0
        ## Generate new parameter to test
        random[4] = random[4]*0.1
        random[5] = random[5]*0.1
        random = np.exp(random)
        new_parameter = np.multiply(parameter,random)
        if Method.method == 'CPZM':
            new_parameter[4] = new_parameter[4]/np.exp(0.1)
        else:
            new_parameter[4] = np.mod(new_parameter[4],2*np.pi) # phi from 0 to 2*pi
        new_parameter[5] = np.mod(new_parameter[5],2*np.pi) # phi from 0 to 2*pi

        ydata_MC = Method.func(xdata, *new_parameter)
        weighted_ydata_MC = np.multiply(weight_array,ydata_MC)
        new_error = np.linalg.norm(weighted_ydata_MC - weighted_ydata)/len(xdata)
        if new_error < error:
            parameter = new_parameter
            error = new_error
   ## If finally gets better fit then plot ##
    if error < error_0:
        stop_MC = False
    if error == error_0:
#        print('Monte Carlo fit get the same fitting parameters')
        stop_MC = True
    return parameter,stop_MC, error
####################################################################
def Fit_Resonator(Resonator,Method):
    manual_init = Method.manual_init
    find_circle = Method.find_circle
    extract_factor = Method.extract_factor
    vary = Method.vary
    filename = Resonator.name
    xdata = Resonator.freq
    ydata = Resonator.S21
    y1data = np.real(ydata)
    y2data = np.imag(ydata)
    x_raw = Resonator.freq
    y1_raw = Resonator.I
    y2_raw = Resonator.Q
    y_raw = Resonator.S21    
## === Step one. Find initial guess automatically and extract part of data only  ====#####
    ## Storage raw data

    init= [0]*6
    if manual_init != None:
        if len(manual_init)==6:
            init = manual_init
            freq = init[3]
            kappa = init[3]/(init[2])
            x_c,y_c,r = 0,0,0
            print("manual initial guess")
        else:
            print(manual_init)
            print("manual input wrong")
    ## Get init from circles
    else:
        if find_circle == True:
            try:
                if Method.method == 'INV':
                    AA = max(abs(1/y_raw))
                    ydata = y_raw*AA
                    y1data = np.real(ydata)
                    y2data = np.imag(ydata)
                    init,x_c,y_c,r = Find_initial_guess(xdata,y1data,y2data,Method,k = 1)
                    init[0] = init[0]*AA
                    x_c = x_c*AA
                    y_c = y_c *AA
                    r = r*AA
                    freq = init[3]
                    kappa = init[3]/(init[2])
                    y1data = np.real(y_raw)
                    y2data = np.imag(y_raw)
                    ydata = y_raw
                else:
                    init,x_c,y_c,r = Find_initial_guess(xdata,y1data,y2data,Method,k = 1)
                    freq = init[3]
                    kappa = init[3]/(init[2])
                    if Method.method == 'CPZM': 
                        kappa = init[3]*init[2]/init[1]
            except:
                try:
                    init,x_c,y_c,r = Find_initial_guess(xdata,y1data,y2data,Method,k = 1)
                    freq = init[3]
                    kappa = init[3]/(init[2])
                    print('1st find circle failed')
                except:
                    print("find initual guess failed. Please manual input guess")
                pass

        else:
            print("find rough guess false")
        ## Extract data near resonate frequency to fit
    if isinstance(extract_factor, list) == True:
        if len(extract_factor ) == 2:
            xstart = extract_factor[0]
            xend = extract_factor[1]
            xdata,y1data = extract_data(x_raw,y1_raw,xstart,xend)
            _,y2data = extract_data(x_raw,y2_raw,xstart,xend)
            _,ydata = extract_data(x_raw,y_raw,xstart,xend)

    elif isinstance(extract_factor, float) == True or isinstance(extract_factor, int) == True:
        
        xstart = freq - extract_factor/2*kappa
        xend = freq + extract_factor/2*kappa
        xdata,y1data = extract_data(x_raw,y1_raw,xstart,xend)
        _,y2data = extract_data(x_raw,y2_raw,xstart,xend)
        _,ydata = extract_data(x_raw,y_raw,xstart,xend)
    elif extract_factor == 'all':
        pass
    else:
        print("extract_factor input failure")
        
    if Method.method == 'INV':
        ydata = ydata**-1 ## Inverse S21
    #####==== Step Two. Fit Both Re and Im data  ====####
        # create a set of Parameters
        ## Monte Carlo Loop for inverse S21
    MC_counts = 0
    error = [10]
    stop_MC = False
    continue_condition = (MC_counts < Method.MC_iteration) and (stop_MC == False)
    output_params = []
    
    
    while continue_condition:
            ## Fit inverse S21 with John Martinis' equation
        params = lmfit.Parameters()
        params.add('amp', value=init[0], vary = vary[0], min=init[0]*0.9 ,max = init[0]*1.1)
        if Method.method == 'DCM':
            params.add('Q', value=init[1],vary = vary[1],min = init[1]*0.5, max = init[1]*1.5)
        elif Method.method == 'INV' or Method.method == 'CPZM':
            params.add('Qi', value=init[1],vary = vary[1],min = init[1]*0.8, max = init[1]*1.2)
        params.add('Qe', value=init[2],vary = vary[2],min = init[2]*0.8, max = init[2]*1.2)
        params.add('w1', value=init[3],vary = vary[3],min = init[3]*0.9, max = init[3]*1.1)
        if Method.method == 'CPZM':
            params.add('Qa', value=init[4], vary = vary[4] , min = init[4]*0.9,max = init[4]*1.1)
        else:
            params.add('phi', value=init[4], vary = vary[4] , min = init[4]*0.9,max = init[4]*1.1)
        params.add('theta', value=init[5], vary = vary[5] , min = init[5]*0.9,max = init[5]*1.1)
        
        if Method.method == 'DCM':
            minner = Minimizer(min_one_Cavity_dip, params, fcn_args=(xdata, ydata))
        elif Method.method == 'INV':
            minner = Minimizer(min_one_Cavity_inverse, params, fcn_args=(xdata, ydata))
        elif Method.method == 'CPZM':
            minner = Minimizer(min_one_Cavity_CPZM, params, fcn_args=(xdata, ydata))

        result = minner.minimize(method = 'least_squares')
        if Method.method == 'CPZM':
            result = minner.minimize(method = 'emcee')
        fit_params = result.params    
        parameter = fit_params.valuesdict()
        
        fit_params = [value for _,value in parameter.items()]
    #####==== Try Monte Carlo Fit Inverse S21 ====####
    
        MC_param,stop_MC, error_MC = \
        MonteCarloFit(xdata,ydata,fit_params,Method) 
        error.append(error_MC)
        if error[MC_counts] < error_MC:
            stop_MC = True
            
        output_params.append(MC_param)
        MC_counts = MC_counts+1
        continue_condition = (MC_counts < Method.MC_iteration) and (stop_MC == False)
        
        if continue_condition == False:
            output_params = output_params[MC_counts-1]
    
    if Method.method == 'DCM':
        figurename =" DCM with Monte Carlo Fit and Raw data\nPower: " + filename
        fig = PlotFit(x_raw,y_raw,output_params,Cavity_DCM,figurename,x_c,y_c,r,extract_factor = [xdata[0],xdata[-1]])
    elif Method.method == 'INV':
        figurename = " Inverse with MC Fit and Raw data\nPower: " + filename
        fig = PlotFit(x_raw,1/y_raw,output_params,Cavity_inverse,figurename,x_c,y_c,r,extract_factor = [xdata[0],xdata[-1]])
    elif Method.method == 'CPZM':
        figurename = " CPZM with MC Fit and Raw data\nPower: " + filename
        fig = PlotFit(x_raw,y_raw,output_params,Cavity_CPZM,figurename,x_c,y_c,r,extract_factor = [xdata[0],xdata[-1]])
    return output_params,fig,min(error),init


###############################################################################
