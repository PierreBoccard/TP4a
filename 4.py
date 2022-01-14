#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:11:03 2021

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
import module as mdl

plt.style.use(astropy_mpl_style)


from IPython import get_ipython

get_ipython().run_line_magic('matplotlib','qt') 

#%%

#-------------- Part 1 --------------
print('------ Part 4.1 ------')
print('\n')

c = 2.998e+5
H0 = 72
omeg_M = 0.3
omeg_lamb = 0.7
K = (H0/c)**2*(omeg_M+omeg_lamb-1)

z = np.linspace(0,3,1001)
I = np.zeros(len(z))
for i in range(len(z)):
    j = quad(mdl.dx12, 0, z[i], args=(c,H0,omeg_M,omeg_lamb))[0]
    I[i] = mdl.dist(j,K,z[i])

def Ez(z):
    return 1/np.sqrt((1+z)**3*omeg_M + omeg_lamb)# + 10**(-5)*(1+z)**2)

DC = np.zeros(len(z))
for i in range(len(z)):
    DC[i] = c/H0 * np.array(quad(Ez, 0, z[i])[0])
    

plt.figure()
plt.plot(z,I/1000,color = 'r',label='Angular Diameter distance')
plt.plot(z,DC/1000,color = 'b',label = 'Comoving distance')
plt.plot(z,(z*c)/H0/1000,color = 'g',label = 'Hubble law')
plt.legend()
plt.ylabel(r'Distance [Gpc]', fontsize = 13)
plt.xlabel(r'Redshift', fontsize = 13)
plt.show()

#%%
Radius = np.array([25.135,19.850]) 
radius = np.array([35.1,30]) 
center = [[821,1136],[819.5,1108]]
pixel = [347]

x_abell = quad(mdl.dx12, 0, 0.375, args=(c,H0,omeg_M,omeg_lamb))[0]
abell_dist = mdl.dist(x_abell,K,0.375)

x_arc = quad(mdl.dx12, 0, 0.72, args=(c,H0,omeg_M,omeg_lamb))[0]
arc_dist = mdl.dist(x_arc,K,0.72)

x_arc_abell = quad(mdl.dx12, 0.375, 0.72, args=(c,H0,omeg_M,omeg_lamb))[0]
arc_abell_dist = mdl.dist(x_arc_abell,K,0.72)

c_Mpc = c * 3.24078e-20
G_Mpc = (3.24078e-23)**3 * 6.67408e-11 / 5.0279e-31
theta = radius * 4.84814e-6
M = mdl.Mass(theta,G_Mpc,abell_dist,arc_dist,arc_abell_dist,c_Mpc)


print('The mass of the cluster is :',"%e"%M[0])
print('\n') 

sig = np.sqrt(theta * c**2 * arc_dist / 4 / np.pi / arc_abell_dist)

print('The velocity dispersion is :',"%e"%sig[0])
print('\n') 

#%%
fileR = './imR.fits'
fileV = './imV.fits'
fileB = './imB.fits'
r = fits.getdata(fileR)
v = fits.getdata(fileV)
b = fits.getdata(fileB)

colorimg = np.zeros([2048,2048,3])
couleur = np.array([r,v,b])
coefficient = np.array([1.3,1.1,2.7])

for i, (coeff,color) in enumerate(zip(coefficient,couleur)):
    colorimg[:,:,i] = coeff*color    


#colorimg2 = colorimg/np.max(colorimg)
vmin, vmax = np.percentile(colorimg,(0.5,99))

colorimg = colorimg - vmin
colorimg = colorimg/vmax

colorimg[colorimg>1] = 1
colorimg[colorimg<0] = 0

plt.figure()
plt.imshow(colorimg)

cropx = 695
cropy = 980
width = 250

colorimg2 = colorimg[900:1600,512:1170,:]
# colorimg2 = colorimg

plt.figure()
plt.imshow(colorimg2)

