from scipy.ndimage import gaussian_filter
import os
import pickle as pkl
import numpy as np  # arrays manipulation
import astropy.io.fits as pyfits  # open / write FITS files
import matplotlib.pyplot as plt  # plots
from PIL import Image  # images manipulation
import math
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
import PyAstronomy as pyasl
import scipy.integrate as integrate
import scipy.special as special
from scipy.interpolate import make_interp_spline, BSpline
import sys
import os
import string
import types
from numpy import *
#plt.style.use(astropy_mpl_style)

import matplotlib.cm as cm
import matplotlib.pyplot as plt  # plots
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.optimize import minimize
import FoFlib as fof

from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','qt') 

slope = np.zeros(17)
slope[16] = 0
slope[15] = 0
# slope[14] = 0
# slope[13] = 0

#%%
plt.figure()
for k in range(4,33,2):
    
    if k < 10:
        
        print(k)
        
        length, off, m, com = fof.read_groups_catalogue('./fof_special_catalogue_00' + str(k))
        pos = fof.read_groups_particles('./fof_special_particles_00'+ str(k))
    
        m = 1e10*m # in msun
        n, b = np.histogram(m, b = np.logspace(np.log10(1e10*142), np.log10(1e10*170000), 50))
        b_mean = [0.5 * (b[i] + b[i+1]) for i in range(len(n))]
        rho = n/((150000)**3)
        
        
        error = np.where(rho <= 0)
        rho = np.delete(rho, error)
        b_mean = np.delete(b_mean, error)
        
        logm = np.log10(b_mean)
        logrho = np.log10(rho)
        pente = np.polyfit(logm,logrho,1) 
        x = np.logspace(12.5, 15.5)
        courbe = 10*(pente[1])*x**(pente[0])
        slope[16-int(k/2)] = pente[0]
        
        plt.loglog(x,courbe,label = 'z = ' + str(int(64-2*k)))
        
    
    if k > 9:
        
        print(k)
        
        length, off, m, com = fof.read_groups_catalogue('./fof_special_catalogue_0' + str(k))
        pos = fof.read_groups_particles('./fof_special_particles_0'+ str(k))
        
        m = 1e10*m # in msun
        n, b = np.histogram(m, b = np.logspace(np.log10(1e10*142), np.log10(1e10*170000), 50))
        b_mean = [0.5 * (b[i] + b[i+1]) for i in range(len(n))]
        rho = n/((150000)**3)
        
        
        error = np.where(rho <= 0)
        rho = np.delete(rho, error)
        b_mean = np.delete(b_mean, error)
        
        logm = np.log10(b_mean)
        logrho = np.log10(rho)
        pente = np.polyfit(logm,logrho,1) 
        x = np.logspace(12.5, 15.5)
        #courbe = 10*(pente[1])*x*(pente[0])
        courbe = 10*(pente[1])*x**(pente[0])
        slope[16-int(k/2)] = pente[0]
        
        
        plt.loglog(x,courbe,label = 'z = ' + str(int(64-2*k)))
        
plt.legend()     
plt.ylabel(r'Density of counts', fontsize = 12)
plt.xlabel(r'Mass of cluster', fontsize = 12)
plt.grid()
plt.show()
#%%

z = np.array([0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,63])

plt.figure()
plt.scatter(z,slope,marker='+',color='k')
plt.legend()
plt.ylabel(r'Slope n', fontsize = 14)
plt.xlabel(r'Redshift z', fontsize = 14)
plt.grid()
plt.show()

#%%

length, off, ma, com = fof.read_groups_catalogue('./fof_special_catalogue_032')

pos = fof.read_groups_particles('./fof_special_particles_032')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#ax.scatter(positions[3000:4000,0],positions[3000:4000,1],positions[3000:4000,2],s=20)
ax.scatter(com[:100,0],com[:100,1],com[:100,2],s=20)
ax.grid()

