#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 14:22:32 2021

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
import module as mdl

plt.style.use(astropy_mpl_style)


from IPython import get_ipython

get_ipython().run_line_magic('matplotlib','qt') 

#%%
#-------------- Part 1 --------------
print('------ Part 1.1 ------')
print('\n')

file = './fiducial.txt'
fiducial = np.loadtxt(file, skiprows=1)

v_fluxid = fiducial[:,0]
vb_fluxid = fiducial[:,1]


plt.figure()
# plt.scatter(vb_fid, v_fid, marker = '.', color = 'r', label = r'Fiducial sequence')
plt.scatter(x = fiducial[:,1], y = fiducial[:,0], s = 85, marker = 'x', color = 'r', label = r'Fiducial sequence')
plt.legend()

#%%

#-------------- Part 2 --------------
print('------ Part 1.3 ------')
print('\n')

file = './V1.txt'
standard_V = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))

file = './B1.txt'
standard_B = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))

C_V = [872.816, 560.113]
C_B = [875.351, 559.559]
A_V = [997.389, 389.573] 
A_B = [999.56, 388.418]
Ru_V = [1116.89, 329.935] 
Ru_B = [1119.06, 329.051]
E_V = [1367.67, 594.833] 
E_B = [1369.54, 594.116]
F_V = [1396.97, 750.001] 
F_B = [1399.23, 748.852]
B_V = [1130.89, 523.889] 
B_B = [1132.63, 522.361]

# data1 = pyfits.getdata('B1.fits')
# data2 = pyfits.getdata('V1.fits')

# x1 = standard_V[:,2]
# y1 = standard_V[:,3]

# plt.figure()
# plt.imshow(data1)

# finding the flux of our known stars (finds the corresponding stars in an interval +-1 in x and y)
C_V_flux = [i[0] for i in standard_V if (i[2]>C_V[0]-1 and i[2]<C_V[0]+1 and i[3]>C_V[1]-1 and i[3]<C_V[1]+1)]
C_B_flux = [i[0] for i in standard_B if (i[2]>C_B[0]-1 and i[2]<C_B[0]+1 and i[3]>C_B[1]-1 and i[3]<C_B[1]+1)]
A_V_flux = [i[0] for i in standard_V if (i[2]>A_V[0]-1 and i[2]<A_V[0]+1 and i[3]>A_V[1]-1 and i[3]<A_V[1]+1)]
A_B_flux = [i[0] for i in standard_B if (i[2]>A_B[0]-1 and i[2]<A_B[0]+1 and i[3]>A_B[1]-1 and i[3]<A_B[1]+1)]
Ru_V_flux = [i[0] for i in standard_V if (i[2]>Ru_V[0]-1 and i[2]<Ru_V[0]+1 and i[3]>Ru_V[1]-1 and i[3]<Ru_V[1]+1)]
Ru_B_flux = [i[0] for i in standard_B if (i[2]>Ru_B[0]-1 and i[2]<Ru_B[0]+1 and i[3]>Ru_B[1]-1 and i[3]<Ru_B[1]+1)]
E_V_flux = [i[0] for i in standard_V if (i[2]>E_V[0]-1 and i[2]<E_V[0]+1 and i[3]>E_V[1]-1 and i[3]<E_V[1]+1)]
E_B_flux = [i[0] for i in standard_B if (i[2]>E_B[0]-1 and i[2]<E_B[0]+1 and i[3]>E_B[1]-1 and i[3]<E_B[1]+1)]
F_V_flux = [i[0] for i in standard_V if (i[2]>F_V[0]-1 and i[2]<F_V[0]+1 and i[3]>F_V[1]-1 and i[3]<F_V[1]+1)]
F_B_flux = [i[0] for i in standard_B if (i[2]>F_B[0]-1 and i[2]<F_B[0]+1 and i[3]>F_B[1]-1 and i[3]<F_B[1]+1)]

total_V = np.array([C_V_flux[0],A_V_flux[0],Ru_V_flux[0],E_V_flux[0],F_V_flux[0]])
total_B = np.array([C_B_flux[0],A_V_flux[0],Ru_V_flux[0],E_V_flux[0],F_V_flux[0]])

# plt.figure()
# plt.scatter(x1,y1)
# plt.scatter(C_V[0],C_V[1],color = 'r')
# plt.scatter(A_V[0],A_V[1],color = 'r')
# plt.scatter(B_V[0],B_V[1],color = 'r')
# plt.scatter(F_V[0],F_V[1],color = 'r')
# plt.scatter(E_V[0],E_V[1],color = 'r')
# plt.scatter(Ru_V[0],Ru_V[1],color = 'r')


magn_theo_V = np.array([12.222,14.341,13.014,12.362,14.564])
magn_theo_B = np.array([-0.013,0.543,-0.19,0.042,0.635]) + magn_theo_V

# constant evaluation
X_b = 1.10900 #  Airmass average for B filter
X_v = 1.10900 #  Airmass average for V filter
b_0 = np.mean(-2.5*np.log10(total_B/2.3926) - magn_theo_B - 0.26*X_b)
v_0 = np.mean(-2.5*np.log10(total_V/0.7377) - magn_theo_V - 0.17*X_v)
b_0_err = np.std(-2.5*np.log10(total_B/2.3926) - magn_theo_B - 0.26*X_b)
v_0_err = np.std(-2.5*np.log10(total_V/0.7377) - magn_theo_V - 0.17*X_v)
#%%
file = './V1f8.txt'
standard_Vf8 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))
file = './B1f8.txt'
standard_Bf8 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))

C_V_flux8 = [i[0] for i in standard_Vf8 if (i[2]>C_V[0]-1 and i[2]<C_V[0]+1 and i[3]>C_V[1]-1 and i[3]<C_V[1]+1)]
C_B_flux8 = [i[0] for i in standard_Bf8 if (i[2]>C_B[0]-1 and i[2]<C_B[0]+1 and i[3]>C_B[1]-1 and i[3]<C_B[1]+1)]
A_V_flux8 = [i[0] for i in standard_Vf8 if (i[2]>A_V[0]-1 and i[2]<A_V[0]+1 and i[3]>A_V[1]-1 and i[3]<A_V[1]+1)]
A_B_flux8 = [i[0] for i in standard_Bf8 if (i[2]>A_B[0]-1 and i[2]<A_B[0]+1 and i[3]>A_B[1]-1 and i[3]<A_B[1]+1)]
Ru_V_flux8 = [i[0] for i in standard_Vf8 if (i[2]>Ru_V[0]-1 and i[2]<Ru_V[0]+1 and i[3]>Ru_V[1]-1 and i[3]<Ru_V[1]+1)]
Ru_B_flux8 = [i[0] for i in standard_Bf8 if (i[2]>Ru_B[0]-1 and i[2]<Ru_B[0]+1 and i[3]>Ru_B[1]-1 and i[3]<Ru_B[1]+1)]
E_V_flux8 = [i[0] for i in standard_Vf8 if (i[2]>E_V[0]-1 and i[2]<E_V[0]+1 and i[3]>E_V[1]-1 and i[3]<E_V[1]+1)]
E_B_flux8 = [i[0] for i in standard_Bf8 if (i[2]>E_B[0]-1 and i[2]<E_B[0]+1 and i[3]>E_B[1]-1 and i[3]<E_B[1]+1)]
F_V_flux8 = [i[0] for i in standard_Vf8 if (i[2]>F_V[0]-1 and i[2]<F_V[0]+1 and i[3]>F_V[1]-1 and i[3]<F_V[1]+1)]
F_B_flux8 = [i[0] for i in standard_Bf8 if (i[2]>F_B[0]-1 and i[2]<F_B[0]+1 and i[3]>F_B[1]-1 and i[3]<F_B[1]+1)]


file = './V1f9.txt'
standard_Vf9 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))
file = './B1f9.txt'
standard_Bf9 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))

C_V_flux9 = [i[0] for i in standard_Vf9 if (i[2]>C_V[0]-1 and i[2]<C_V[0]+1 and i[3]>C_V[1]-1 and i[3]<C_V[1]+1)]
C_B_flux9 = [i[0] for i in standard_Bf9 if (i[2]>C_B[0]-1 and i[2]<C_B[0]+1 and i[3]>C_B[1]-1 and i[3]<C_B[1]+1)]
A_V_flux9 = [i[0] for i in standard_Vf9 if (i[2]>A_V[0]-1 and i[2]<A_V[0]+1 and i[3]>A_V[1]-1 and i[3]<A_V[1]+1)]
A_B_flux9 = [i[0] for i in standard_Bf9 if (i[2]>A_B[0]-1 and i[2]<A_B[0]+1 and i[3]>A_B[1]-1 and i[3]<A_B[1]+1)]
Ru_V_flux9 = [i[0] for i in standard_Vf9 if (i[2]>Ru_V[0]-1 and i[2]<Ru_V[0]+1 and i[3]>Ru_V[1]-1 and i[3]<Ru_V[1]+1)]
Ru_B_flux9 = [i[0] for i in standard_Bf9 if (i[2]>Ru_B[0]-1 and i[2]<Ru_B[0]+1 and i[3]>Ru_B[1]-1 and i[3]<Ru_B[1]+1)]
E_V_flux9 = [i[0] for i in standard_Vf9 if (i[2]>E_V[0]-1 and i[2]<E_V[0]+1 and i[3]>E_V[1]-1 and i[3]<E_V[1]+1)]
E_B_flux9 = [i[0] for i in standard_Bf9 if (i[2]>E_B[0]-1 and i[2]<E_B[0]+1 and i[3]>E_B[1]-1 and i[3]<E_B[1]+1)]
F_V_flux9 = [i[0] for i in standard_Vf9 if (i[2]>F_V[0]-1 and i[2]<F_V[0]+1 and i[3]>F_V[1]-1 and i[3]<F_V[1]+1)]
F_B_flux9 = [i[0] for i in standard_Bf9 if (i[2]>F_B[0]-1 and i[2]<F_B[0]+1 and i[3]>F_B[1]-1 and i[3]<F_B[1]+1)]


file = './V1f10.txt'
standard_Vf10 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))
file = './B1f10.txt'
standard_Bf10 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))

C_V_flux10 = [i[0] for i in standard_Vf10 if (i[2]>C_V[0]-1 and i[2]<C_V[0]+1 and i[3]>C_V[1]-1 and i[3]<C_V[1]+1)]
C_B_flux10 = [i[0] for i in standard_Bf10 if (i[2]>C_B[0]-1 and i[2]<C_B[0]+1 and i[3]>C_B[1]-1 and i[3]<C_B[1]+1)]
A_V_flux10 = [i[0] for i in standard_Vf10 if (i[2]>A_V[0]-1 and i[2]<A_V[0]+1 and i[3]>A_V[1]-1 and i[3]<A_V[1]+1)]
A_B_flux10 = [i[0] for i in standard_Bf10 if (i[2]>A_B[0]-1 and i[2]<A_B[0]+1 and i[3]>A_B[1]-1 and i[3]<A_B[1]+1)]
Ru_V_flux10 = [i[0] for i in standard_Vf10 if (i[2]>Ru_V[0]-1 and i[2]<Ru_V[0]+1 and i[3]>Ru_V[1]-1 and i[3]<Ru_V[1]+1)]
Ru_B_flux10 = [i[0] for i in standard_Bf10 if (i[2]>Ru_B[0]-1 and i[2]<Ru_B[0]+1 and i[3]>Ru_B[1]-1 and i[3]<Ru_B[1]+1)]
E_V_flux10 = [i[0] for i in standard_Vf10 if (i[2]>E_V[0]-1 and i[2]<E_V[0]+1 and i[3]>E_V[1]-1 and i[3]<E_V[1]+1)]
E_B_flux10 = [i[0] for i in standard_Bf10 if (i[2]>E_B[0]-1 and i[2]<E_B[0]+1 and i[3]>E_B[1]-1 and i[3]<E_B[1]+1)]
F_V_flux10 = [i[0] for i in standard_Vf10 if (i[2]>F_V[0]-1 and i[2]<F_V[0]+1 and i[3]>F_V[1]-1 and i[3]<F_V[1]+1)]
F_B_flux10 = [i[0] for i in standard_Bf10 if (i[2]>F_B[0]-1 and i[2]<F_B[0]+1 and i[3]>F_B[1]-1 and i[3]<F_B[1]+1)]


file = './V1f15.txt'
standard_Vf15 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))
file = './B1f15.txt'
standard_Bf15 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))

C_V_flux15 = [i[0] for i in standard_Vf15 if (i[2]>C_V[0]-1 and i[2]<C_V[0]+1 and i[3]>C_V[1]-1 and i[3]<C_V[1]+1)]
C_B_flux15 = [i[0] for i in standard_Bf15 if (i[2]>C_B[0]-1 and i[2]<C_B[0]+1 and i[3]>C_B[1]-1 and i[3]<C_B[1]+1)]
A_V_flux15 = [i[0] for i in standard_Vf15 if (i[2]>A_V[0]-1 and i[2]<A_V[0]+1 and i[3]>A_V[1]-1 and i[3]<A_V[1]+1)]
A_B_flux15 = [i[0] for i in standard_Bf15 if (i[2]>A_B[0]-1 and i[2]<A_B[0]+1 and i[3]>A_B[1]-1 and i[3]<A_B[1]+1)]
Ru_V_flux15 = [i[0] for i in standard_Vf15 if (i[2]>Ru_V[0]-1 and i[2]<Ru_V[0]+1 and i[3]>Ru_V[1]-1 and i[3]<Ru_V[1]+1)]
Ru_B_flux15 = [i[0] for i in standard_Bf15 if (i[2]>Ru_B[0]-1 and i[2]<Ru_B[0]+1 and i[3]>Ru_B[1]-1 and i[3]<Ru_B[1]+1)]
E_V_flux15 = [i[0] for i in standard_Vf15 if (i[2]>E_V[0]-1 and i[2]<E_V[0]+1 and i[3]>E_V[1]-1 and i[3]<E_V[1]+1)]
E_B_flux15 = [i[0] for i in standard_Bf15 if (i[2]>E_B[0]-1 and i[2]<E_B[0]+1 and i[3]>E_B[1]-1 and i[3]<E_B[1]+1)]
F_V_flux15 = [i[0] for i in standard_Vf15 if (i[2]>F_V[0]-1 and i[2]<F_V[0]+1 and i[3]>F_V[1]-1 and i[3]<F_V[1]+1)]
F_B_flux15 = [i[0] for i in standard_Bf15 if (i[2]>F_B[0]-1 and i[2]<F_B[0]+1 and i[3]>F_B[1]-1 and i[3]<F_B[1]+1)]

file = './V1f20.txt'
standard_Vf20 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))
file = './B1f20.txt'
standard_Bf20 = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))

C_V_flux20 = [i[0] for i in standard_Vf20 if (i[2]>C_V[0]-1 and i[2]<C_V[0]+1 and i[3]>C_V[1]-1 and i[3]<C_V[1]+1)]
C_B_flux20 = [i[0] for i in standard_Bf20 if (i[2]>C_B[0]-1 and i[2]<C_B[0]+1 and i[3]>C_B[1]-1 and i[3]<C_B[1]+1)]
A_V_flux20 = [i[0] for i in standard_Vf20 if (i[2]>A_V[0]-1 and i[2]<A_V[0]+1 and i[3]>A_V[1]-1 and i[3]<A_V[1]+1)]
A_B_flux20 = [i[0] for i in standard_Bf20 if (i[2]>A_B[0]-1 and i[2]<A_B[0]+1 and i[3]>A_B[1]-1 and i[3]<A_B[1]+1)]
Ru_V_flux20 = [i[0] for i in standard_Vf20 if (i[2]>Ru_V[0]-1 and i[2]<Ru_V[0]+1 and i[3]>Ru_V[1]-1 and i[3]<Ru_V[1]+1)]
Ru_B_flux20 = [i[0] for i in standard_Bf20 if (i[2]>Ru_B[0]-1 and i[2]<Ru_B[0]+1 and i[3]>Ru_B[1]-1 and i[3]<Ru_B[1]+1)]
E_V_flux20 = [i[0] for i in standard_Vf20 if (i[2]>E_V[0]-1 and i[2]<E_V[0]+1 and i[3]>E_V[1]-1 and i[3]<E_V[1]+1)]
E_B_flux20 = [i[0] for i in standard_Bf20 if (i[2]>E_B[0]-1 and i[2]<E_B[0]+1 and i[3]>E_B[1]-1 and i[3]<E_B[1]+1)]
F_V_flux20 = [i[0] for i in standard_Vf20 if (i[2]>F_V[0]-1 and i[2]<F_V[0]+1 and i[3]>F_V[1]-1 and i[3]<F_V[1]+1)]
F_B_flux20 = [i[0] for i in standard_Bf20 if (i[2]>F_B[0]-1 and i[2]<F_B[0]+1 and i[3]>F_B[1]-1 and i[3]<F_B[1]+1)]


xB = np.array([5,8,9,10,15,20])

yBA = np.array([A_B_flux,A_B_flux8,A_B_flux9,A_B_flux10,A_B_flux15,A_B_flux20])
yBA = yBA/np.max(yBA)
yVA = np.array([A_V_flux,A_V_flux8,A_V_flux9,A_V_flux10,A_V_flux15,A_V_flux20])
yVA = yVA/np.max(yVA)

yBE = np.array([E_B_flux,E_B_flux8,E_B_flux9,E_B_flux10,E_B_flux15,E_B_flux20])
yBE = yBE/np.max(yBE)
yVE = np.array([E_V_flux,E_V_flux8,E_V_flux9,E_V_flux10,E_V_flux15,E_V_flux20])
yVE = yVE/np.max(yVE)

yBRu = np.array([Ru_B_flux,Ru_B_flux8,Ru_B_flux9,Ru_B_flux10,Ru_B_flux15,Ru_B_flux20])
yBRu = yBRu/np.max(yBRu)
yVRu = np.array([Ru_V_flux,Ru_V_flux8,Ru_V_flux9,Ru_V_flux10,Ru_V_flux15,Ru_V_flux20])
yVRu = yVRu/np.max(yVRu)

yBC = np.array([C_B_flux,C_B_flux8,C_B_flux9,C_B_flux10,C_B_flux15,C_B_flux20])
yBC = yBC/np.max(yBC)
yVC = np.array([C_V_flux,C_V_flux8,C_V_flux9,C_V_flux10,C_V_flux15,C_V_flux20])
yVC = yVC/np.max(yVC)

yBF = np.array([F_B_flux,F_B_flux8,F_B_flux9,F_B_flux10,F_B_flux15,F_B_flux20])
yBF = yBF/np.max(yBF)
yVF = np.array([F_V_flux,F_V_flux8,F_V_flux9,F_V_flux10,F_V_flux15,F_V_flux20])
yVF = yVF/np.max(yVF)

#%%

plt.figure()
plt.plot(xB,yBA,color = 'blue')
plt.plot(xB,yVA,color = 'green')
plt.plot(xB,yBE,color = 'red')
plt.plot(xB,yVE,color = 'black')
plt.plot(xB,yBF,color = 'pink')
plt.plot(xB,yVF,color = 'orange')
plt.plot(xB,yBRu,color = 'violet')
plt.plot(xB,yVRu,color = 'brown')
plt.plot(xB,yBC,color = 'magenta')
plt.plot(xB,yVC,color = 'turquoise')

plt.figure()
plt.plot(xB,(yBA + yBE + yBF + yBRu + yBC)/5,color = 'blue',label = 'Blue filter')
plt.plot(xB,(yVA + yVE + yVF + yVRu + yVC)/5,color = 'green',label = 'Green filter')
plt.legend()
plt.xlabel(r'Radius of the aperture in pixel',fontsize=16)
plt.ylabel(r'Fraction of total flux collected',fontsize=16)


#%%

ratio_A_V = A_V_flux20[0]/A_V_flux[0]
ratio_A_B = A_B_flux20[0]/A_B_flux[0]

ratio_E_V = E_V_flux20[0]/E_V_flux[0]
ratio_E_B = E_B_flux20[0]/E_B_flux[0]

ratio_Ru_V = Ru_V_flux20[0]/Ru_V_flux[0]
ratio_Ru_B = Ru_B_flux20[0]/Ru_B_flux[0]

ratio_C_V = C_V_flux20[0]/C_V_flux[0]
ratio_C_B = C_B_flux20[0]/C_B_flux[0]

ratio_F_V = F_V_flux20[0]/F_V_flux[0]
ratio_F_B = F_B_flux20[0]/F_B_flux[0]

xbis = np.array([1,2,3,4,5])
ratioB = np.array([ratio_A_B,ratio_E_B,ratio_Ru_B,ratio_F_B,ratio_C_B])
ratioV = np.array([ratio_A_V,ratio_E_V,ratio_Ru_V,ratio_F_V,ratio_C_V])

ratioBfinal = np.mean(ratioB)
ratioVfinal = np.mean(ratioV)


#%%

#-------------- Part 1 bis --------------
print('------ Part 1.2 ------')
print('\n')

param_number = 5   # number of parameters loaded into the file

# loading of parameters for green filter
file = './V.txt'
tableau_V = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))
t_V = 0.7353
X_V = 2.05000

#tableau_V[:,0] = tableau_V[:,0]*ratioVfinal

# loading of parameters for blue filter
file = './B.txt'
tableau_B = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))
t_B = 2.5026
X_B = 2.06000

file = './V20.txt'
tableau_Vlong = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))
t_Vlong = 20.0055
X_Vlong = 1.81900

#tableau_V[:,0] = tableau_V[:,0]*ratioVfinal

# loading of parameters for blue filter
file = './B60.txt'
tableau_Blong = np.loadtxt(file, skiprows=0, usecols=(0,1,2,3,4))
t_Blong = 59.9927
X_Blong = 1.80300

#tableau_B[:,0] = tableau_B[:,0]*ratioBfinal

magnitude_B, magnitude_V, s_V, s_B = mdl.align_im(tableau_V,tableau_B,0,0,5,X_V,X_B,-27.8,-27.1,t_V,t_B,1,1)  # see tree module for more info
# magnitude_B, magnitude_V, s_V, s_B = mdl.align_im(tableau_V,tableau_B,0,0,5,X_V,X_B,v_0,b_0,t_V,t_B,1,1)  # see tree module for more info

magnitude_Blong, magnitude_Vlong, s_Vlong, s_Blong = mdl.align_im(tableau_Vlong,tableau_Blong,0,0,5,X_Vlong,X_Blong,-27.5,-26.9,t_Vlong,t_Blong,1,1)  # see tree module for more info
#magnitude_Blong, magnitude_Vlong, s_Vlong, s_Blong = mdl.align_im(tableau_Vlong,tableau_Blong,0,0,5,X_Vlong,X_Blong,b_0,v_0,t_Vlong,t_Blong,1,1)  # see tree module for more info

plt.figure()
plt.scatter(magnitude_B-magnitude_V,magnitude_V, s = 5, marker = 'x', color = "k", label = r'Short exposure time ')
#plt.scatter(magnitude_Blong-magnitude_Vlong,magnitude_Vlong, s = 10, marker = '.', color = "k",label = r'Long exposure time')
plt.scatter(x = fiducial[:,1], y = fiducial[:,0], s = 25, marker = 'x', color = 'r', label = r'Fiducial sequence')
plt.gca().invert_yaxis()
plt.legend()
plt.xlabel('Apparent magnitude $m_B - m_V$',fontsize=16)
plt.ylabel('Apparent magnitude $m_V$',fontsize=16)
#matplotlib.pyplot.arrow(1.275, , dx, dy)
plt.show()
#%%

fname = './v85iso.txt'
f = np.loadtxt(fname, skiprows=0, usecols=(0,1,2,3,4,5,6,7,8,9,10,11))

Mv_age0 = [i[8] for i in f if (i[0]==0.2)]
B_V_age0 = [i[9] for i in f if (i[0]==0.2)]

fname = './vb85iso.txt'
f = np.loadtxt(fname, skiprows=0, usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23))

Mv_age02 = np.array([[i[2],i[8],i[9]] for i in f if ((i[0]==0.2) and (i[1]==1e-4))])
Mv_age02bis = np.array([[i[2],i[8],i[9]] for i in f if ((i[0]==0.3) and (i[1]==1e-4))])
#B_V_age02 = [i[9] for i in f if (i[0]==0.2)]]

#%%

plt.figure()
plt.scatter(magnitude_B-magnitude_V,magnitude_V, s = 10, marker = '.',color = 'grey',label = r'Measured data') 
plt.scatter(x = fiducial[:,1], y = fiducial[:,0], s = 35, marker = 'x', color = 'k', label = r'Fiducial sequence')
for i in np.unique(Mv_age02[:,0]):
    if i == 14:
        bv = np.array([j[2] for j in Mv_age02 if j[0]==i]) + 0.15 #+ 5*np.log(100) - 5
        v = np.array([j[1] for j in Mv_age02 if j[0]==i]) + 5*np.log(60) - 5
        plt.plot(bv,v,color = "r",label = 'Isochrone of age 14 Gyr',linewidth=2)
for i in np.unique(Mv_age02[:,0]):
    if i == 16:
        bv = np.array([j[2] for j in Mv_age02 if j[0]==i]) + 0.15 #+ 5*np.log(100) - 5
        v = np.array([j[1] for j in Mv_age02 if j[0]==i]) + 5*np.log(60) - 5
        plt.plot(bv,v,color = "blue",label = 'Isochrone of age 16 Gyr',linewidth=2)        
        #plt.scatter(magnitude_B-magnitude_V,magnitude_V, s = 10, marker = '.') 
        #plt.scatter(x = fiducial[:,1], y = fiducial[:,0], s = 25, marker = 'x', color = 'r', label = r'Fiducial sequence')
plt.xlabel('Apparent magnitude $m_B - m_V$',fontsize=16)
plt.ylabel('Apparent magnitude $m_V$',fontsize=16)
plt.gca().invert_yaxis()
plt.legend()
#%%

plt.figure()
#plt.scatter(magnitude_B-magnitude_V,magnitude_V, s = 10, marker = '.',color = 'grey',label = r'Measured data') 
plt.scatter(x = fiducial[:,1], y = fiducial[:,0], s = 35, marker = 'x', color = 'k', label = r'Fiducial sequence')
for i in np.unique(Mv_age02[:,0]):
    if i == 12:
        bv = np.array([j[2] for j in Mv_age02bis if j[0]==i])+0.15  #+ 5*np.log(100) - 5
        v = np.array([j[1] for j in Mv_age02bis if j[0]==i]) + 5*np.log(57) - 5
        plt.plot(bv,v,color = "r",label = 'Isochrone',linewidth=2)
plt.xlabel('Apparent magnitude $m_B - m_V$',fontsize=16)
plt.ylabel('Apparent magnitude $m_V$',fontsize=16)
plt.gca().invert_yaxis()
plt.legend()

