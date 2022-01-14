#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 10:02:10 2021

@author: PierreBoccard
"""

from scipy.ndimage import gaussian_filter
import os
import pickle
import numpy as np  # arrays manipulation
import astropy.io.fits as pyfits  # open / write FITS files
import matplotlib.pyplot as plt  # plots
from PIL import Image  # images manipulation
import math
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.optimize import leastsq

# estimation of quantities
M_disk = 10**(10)    
M_cm = 0.8*10**(7)
a =  2   # disk scale length
r_0 = 1   # halo scale length
rho_0 = 7.2*10**(6)    # halo density
G = 6.67*10**(-11)*(3.2408e-20)**3 * (2*10**(30))
D = 9200


# definition of the disk potential, M_d is the mass of the disk (by default set at Milky Way),
# sl_d the scale length (by default set at Milky Way), r is the distance
# and C a constant add
def disk_potential(r, C, M_d = M_disk, sl_d = a):
    return -G*M_d/(r**2+sl_d**2)**(1/2) + C
 
    
# definition of the dark matter halo potential, M_h is the mass of the halo (by default set at Milky Way),
# sl_h the scale length (by default set at Milky Way), r is the distance,
# rho is the mass density and C a constant add
def halo_potential(r, C, sl_h = r_0, rho = rho_0):
    return 4*np.pi*G*rho*sl_h**2 * (0.5*np.log(1+(r/sl_h)**2)+(sl_h/r)*np.arctan(r/sl_h)) + C


# definition of the disk potential, M_d is the mass of the buldge (by default set at Milky Way),
# r is the distance and C a constant add
def CM_potential(r, C, M_c = M_cm):
    return -G*M_cm/r + C

    
def V_disk(r, M_d, sl_d):
    return np.sqrt(abs(r**2 * G * M_d / (r**2 + sl_d**2)**(3/2)))*3.086e+16
    
    
def V_halo(r, sl_h, rho):
    return np.sqrt(abs(4*np.pi*G*rho*sl_h**2 * ( 1 - (sl_h/r) * np.arctan(r/sl_h) )))*3.086e+16
    
    
def V_CM(r, M_c):
    return np.sqrt(G * M_c / r)*3.086e+16
    
    
def V_tot(r, M_c, M_d, sl_d, sl_h, rho):
    return np.sqrt(V_disk(r,M_d,sl_d)**2 + V_halo(r,sl_h,rho)**2 + V_CM(r,M_c)**2)

