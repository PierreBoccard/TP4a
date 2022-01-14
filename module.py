#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:33:01 2021

@author: PierreBoccard
"""


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
from scipy.optimize import leastsq
from scipy.integrate import quad

plt.style.use(astropy_mpl_style)


from IPython import get_ipython

get_ipython().run_line_magic('matplotlib','qt') 

#%%

def dx12(z, c, H0, omeg_M, omeg_lamb):
    return (c/H0/(np.sqrt((1-omeg_M-omeg_lamb)*(1+z)**2 + omeg_M*(1+z)**3 + omeg_lamb)))

def Hubble(z, c, H0):
    return (((z+1)**2-1)/((z+1)**2+1)*c/H0)

def dist(x, K, z):
    if K == 0:
        return (x/(1+z))
    return (1/(1+z)*(1/np.sqrt(abs(K))*np.sin(np.sqrt(abs(K))*x)))

def Mass(theta, G, Dd, Ds, Dds, c):
    return (theta**2 * c**2 * Dd * Ds / (4*G*Dds))

def Ez(z,omeg_M, omeg_lamb):
    return 1/np.sqrt((1+z**3)*omeg_M + omeg_lamb + 10**(-5)*(1+z)**2)