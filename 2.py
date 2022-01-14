#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:36:51 2021

@author: PierreBoccard
"""

#!/usr/bin/env python3

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

#plt.style.use(astropy_mpl_style)


from IPython import get_ipython

import sys
sys.path.append("./1_NGC3198")
import velocity as vl

#-------------- Part 1 --------------
# take care that units are kpc and solar masses when no
# other indications are given
# Velocities from vl are directly converted in km/s

# estimation of quantities
M_disk = 10**(10)    
M_cm = 0.8*10**(9)
a =  2    # disk scale length
r_0 = 1   # halo scale length
rho_0 = 7.2*10**(6)    # halo density === 2 × 10-25 g cm-3 
G = 6.67*10**(-11)*(3.2408e-20)**3 * (2*10**(30))
D = 9200
c = 299792458   # speed of light in m/s

# function that returns a convolution with a gaussian function
def convolution_gauss(S,sigma):
    return gaussian_filter(S,sigma)

# inclinaison in degrees
i = 75  #rough estimation

# loading of the file
fname = 'ngc3198_rot_curve.txt'
velocity = np.loadtxt(fname, skiprows=2, usecols=0)*np.sin(np.pi*i/180)
v_err = np.loadtxt(fname, skiprows=2, usecols=1)*np.sin(np.pi*i/180)
r = np.loadtxt(fname, skiprows=2, usecols=2)* D * 0.00029088599995183


get_ipython().run_line_magic('matplotlib','qt') 


# # #--------------------------------------------------------


print('------ Part 2.1.2 ------')
print('\n')
print('------ The following values are obtained using estimations of quantities ------')
print('The parameters value are :')
print('\n')
print('Mass of the buldge : ', M_cm/10000000000)
print('\n')
print('Mass of the disk : ', M_disk/100000000)
print('\n')
print('Scale length of the disk : ', a)
print('\n')
print('Scale length of the halo : ', r_0)
print('\n')
print('Density of the halo : ', rho_0/1000000)
print('\n')


#define spline
rnew = np.linspace(r.min(), r.max(), 200) 
spl = make_interp_spline(r, velocity, k=1)
velocity_smooth = spl(rnew)


# plot of the 4 velocities with estimated  parameters 
plt.figure()
plt.errorbar(r,velocity,v_err, color = 'k',label = 'Circular velocity measured')
plt.plot(r,vl.V_tot(r, M_cm, M_disk, a, r_0, rho_0), '-r', label='Total circular velocity computed')
plt.plot(r,vl.V_disk(r, M_disk, a), 'b', label='Circular velocity of disk')
plt.plot(r,vl.V_halo(r, a, rho_0), color='orange', label='Circular velocity of halo')
plt.plot(r,vl.V_CM(r, M_cm), 'g', label='Circular velocity of bulge')
plt.grid()
#plt.errorbar(rnew,velocity_smooth,v_err, color = 'k')
#plt.plot(r,vl.V_tot(r, M_cm, M_disk, a, r_0, rho_0),r,vl.V_disk(r, M_disk, a),r,vl.V_halo(r, a, rho_0),r,vl.V_CM(r, M_cm))

# Add title and axis names
plt.legend()
plt.xlabel(r'Distance from galaxy center r [kpc]',fontsize=16)
plt.ylabel(r'Circular velocity [kms$^{-1}$]',fontsize=16)


#%%
# # --------------------------------------------------------

# curve fitting
p0 = np.array([M_cm,M_disk,a,r_0,rho_0])
poptR, pcov = curve_fit(vl.V_tot, r, velocity, p0, v_err)
perr = np.sqrt(np.diag(pcov))


print('------ Part 2.1.3 ------')
print('\n')
print('------ The following values are obtained using gradient descent optimiser ------')
print('\n')
print('The parameters value are :')
print('\n')
print('Mass of the buldge :', poptR[0]/100000000, '+-', perr[0]//100000000)
print('\n')
print('Mass of the disk :', poptR[1]/10000000000, '+-', perr[1]/10000000000)
print('\n')
print('Scale length of the disk :', poptR[2], '+-', "%e"%perr[2])
print('\n')
print('Scale length of the halo :', poptR[3], '+-', "%e"%perr[3])
print('\n')
print('Density of the halo :',  poptR[4]/1000000, '+-',  perr[4]/1000000)
print('\n')

# plot of the 4 velocities with estimated  parameters 
plt.figure()
plt.plot(r,vl.V_tot(r, poptR[0], poptR[1], poptR[2], poptR[3], poptR[4]), '-r', label='Total circular velocity computed')
plt.plot(r,vl.V_disk(r, poptR[1], poptR[2]), '-b', label='Circular velocity of disk')
plt.plot(r,vl.V_halo(r, poptR[3], poptR[4]), color='orange', label='Circular velocity of halo')
plt.plot(r,vl.V_CM(r, poptR[0]), '-g', label='Circular velocity of bulge')
plt.errorbar(r,velocity,v_err, color = 'k', label = 'Circular velocity measured')
plt.grid()
# plt.plot(r,vl.V_tot(r, poptR[0], poptR[1], poptR[2], poptR[3], poptR[4]),r,vl.V_disk(r, poptR[1], poptR[2]),r,vl.V_halo(r, poptR[2], poptR[4]),r,vl.V_CM(r, poptR[0]))

# Add title and axis names
plt.legend(loc = (0.48,0.5))
plt.xlabel(r'Distance from galaxy center r [kpc]', fontsize = 13)
plt.ylabel(r'Circular velocity [kms$^{-1}$]', fontsize = 13)


#%% # --------------------------------------------------------


# curve fitting
p0 = np.array([M_cm,M_disk,a,r_0,rho_0])
poptR, pcov = curve_fit(vl.V_tot, r, velocity, p0, v_err)
perr = np.sqrt(np.diag(pcov))

print('------ Part 2.1.4 ------')
print('\n')
print('------ The following values are obtained using Markov-Chain Monte-Carlo ------')
print('\n')
print('The parameters value are :')
print('\n')
print('Mass of the buldge :', poptR[0]/100000000, '+-', perr[0]//100000000)
print('\n')
print('Mass of the disk :', poptR[1]/10000000000, '+-', perr[1]/10000000000)
print('\n')
print('Scale length of the disk :', poptR[2], '+-', "%e"%perr[2])
print('\n')
print('Scale length of the halo :', poptR[3], '+-', "%e"%perr[3])
print('\n')
print('Density of the halo :',  poptR[4]/1000000, '+-',  perr[4]/1000000)
print('\n')

# loading of LL and parameters
L = pkl.load(open('./logLikelihood.pkl', 'rb'))
ind = np.argmax(L)   # index of the max of LL
chain_fit = pkl.load(open('./samples.pkl', 'rb'))
chain_fit = chain_fit.T

p1, p2, p3, p4, p5 = chain_fit[1][ind], chain_fit[2][ind], 10**(chain_fit[3][ind]), 10**(chain_fit[4][ind]), 10**(chain_fit[0][ind])    # parameters that maximize the LL


# plot of the 4 velocities with parameters maximizing LL
plt.figure()
plt.plot(r,vl.V_tot(r,p4,p5,p1,p2,p3), '-r', label='Total circular velocity computed')
plt.plot(r,vl.V_disk(r,p5,p1), '-b', label='Circular velocity of disk')
plt.plot(r,vl.V_halo(r,p2,p3), color='orange', label='Circular velocity of halo')
plt.plot(r,vl.V_CM(r,p4), '-g', label='Circular velocity of bulge')
plt.errorbar(r,velocity,v_err, color = 'k', label = 'Circular velocity measured')
plt.grid()
# Add title and axis names
plt.legend(loc = (0.45,0.2))
plt.xlabel('Distance from galaxy center [kpc]')
plt.ylabel('Circular velocity [kms$^{-1}$]')


print('The following parameters maximize the LL function under restrictions (see MCMC script) : ')
print('M_cm =', p4/100000000)
print('\n')
print('M_disk =', p5/10000000000)
print('\n')
print('a =', p1)
print('\n')
print('r_0 =', p2)
print('\n')
print('rho_0 =', p3/1000000)
print('\n')


### -----------------------------------------
#%%

k = 0
Vecteur_G = np.zeros(25)
ErrVecteur_G = np.zeros(25)

for j in range(1, 100, 4):
    M = np.zeros(1000)
    
    for i in range(0,999):
        N = len(chain_fit[0])
        p1, p2, p3, p4, p5 = chain_fit[1][N-(i+1)], chain_fit[2][N-(i+1)], 10**(chain_fit[3][N-(i+1)]), 10**(chain_fit[4][N-(i+1)]), 10**(chain_fit[0][N-(i+1)])
        M[i] = j/G*(vl.V_tot(j,p4,p5,p1,p2,p3)*3.24078e-17)**2     
    
    disp = np.std(M) 
    Vecteur_G[k] = np.mean(M)
    ErrVecteur_G[k] = np.std(M) 
    k = k+1
        
    
print('The mass of the galaxy is :', "%e"%np.mean(M), '+-', "%e"%disp)
print('\n') 
        
    #%%

### -----------------------------------------

k = 0
Vecteur_H = np.zeros(25)
ErrVecteur_H = np.zeros(25)
for j in range(1, 100, 4):
    M = np.zeros(1000)
    
    for i in range(0,999):
        N = len(chain_fit[0])
        p1, p2, p3, p4, p5 = chain_fit[1][N-(i+1)], chain_fit[2][N-(i+1)], 10**(chain_fit[3][N-(i+1)]), 10**(chain_fit[4][N-(i+1)]), 10**(chain_fit[0][N-(i+1)])
        M[i] = j/G*(vl.V_halo(j,p2,p3)*3.24078e-17)**2       
    
    disp = np.std(M)      
    Vecteur_H[k] = np.mean(M)
    ErrVecteur_H[k] = np.std(M) 
    k = k+1
    
print('The mass of the halo is :', "%e"%np.mean(M), '+-', "%e"%disp)
print('\n') 

#%%

### -----------------------------------------

k = 0
Vecteur_D = np.zeros(25)
ErrVecteur_D = np.zeros(25)
for j in range(1, 100, 4):
    M = np.zeros(1000)
    
    for i in range(0,999):
        N = len(chain_fit[0])
        p1, p2, p3, p4, p5 = chain_fit[1][N-(i+1)], chain_fit[2][N-(i+1)], 10**(chain_fit[3][N-(i+1)]), 10**(chain_fit[4][N-(i+1)]), 10**(chain_fit[0][N-(i+1)])
        M[i] = j/G*(vl.V_disk(j,p5,p1)*3.24078e-17)**2        
    
    disp = np.std(M)      
    Vecteur_D[k] = np.mean(M)
    ErrVecteur_D[k] = np.std(M) 
    k = k+1
                   
print('The mass of the disk is :', "%e"%np.mean(M), '+-', "%e"%disp)
print('\n') 

#%%

### -----------------------------------------

k = 0
Vecteur_B = np.zeros(25)
ErrVecteur_B = np.zeros(25)
for j in range(1, 100, 4):
    M = np.zeros(1000)
    
    for i in range(0,999):
        N = len(chain_fit[0])
        p1, p2, p3, p4, p5 = chain_fit[1][N-(i+1)], chain_fit[2][N-(i+1)], 10**(chain_fit[3][N-(i+1)]), 10**(chain_fit[4][N-(i+1)]), 10**(chain_fit[0][N-(i+1)])
        M[i] = j/G*(vl.V_CM(j,p4)*3.24078e-17)**2        
    
    disp = np.std(M)      
    Vecteur_B[k] = np.mean(M)
    ErrVecteur_B[k] = np.std(M) 
    k = k+1 
        
print('The mass of the bulge is :', "%e"%np.mean(M), '+-', "%e"%disp)
print('\n') 


# # # --------------------------------------------------------

#%%

x = np.array([1,4,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69,73,77,81,85,89,93,97])

plt.figure()
plt.errorbar(x,Vecteur_G,yerr = ErrVecteur_G,color = 'r',fmt='x',capsize=6,label = 'Total mass')
plt.errorbar(x,Vecteur_H,yerr = ErrVecteur_H,color = 'b',fmt='x',capsize=6,label = 'Halo mass')
plt.errorbar(x,Vecteur_D,yerr = ErrVecteur_D,color = 'g',fmt='x',capsize=6,label = 'Disk mass ')
plt.errorbar(x,Vecteur_B,yerr = ErrVecteur_B,fmt='x',capsize=6,label = '')
# plt.scatter(x,Vecteur_G,marker='+',color='k',label='Total mass')
# plt.scatter(x,Vecteur_H,marker='+',color='g',label='Halo mass')
# plt.scatter(x,Vecteur_D,marker='+',color='b',label='Disk mass')
#plt.scatter(x,Vecteur_B,marker='+',color='r',label='Bulge mass')
plt.grid()
plt.legend(loc = (0.45,0.2))
plt.xlabel('Radius [kpc]')
plt.ylabel('Mass inside r [$M_{\odot}$]')
plt.legend()


#%%

# # #-------------- Part 2 --------------    
print('------ Part 2.2 ------')
print('\n') 
    
# loading of the spectrums
file_name = "./k_star.fits"
rawdata1 = pyfits.open(file_name) 
k_spectrum = rawdata1[0].data 

file_name = "./elliptical_galaxy.fits"
rawdata2 = pyfits.open(file_name) 
galaxy_spectrum = rawdata2[0].data   

# quantities extracted from the header of the spectrums

hd1 = rawdata1[0].header
crval1 = hd1['CRVAL1']
cdelt1 = hd1['CDELT1']

hd2 = rawdata2[0].header
crval2 = hd2['CRVAL1']
cdelt2 = hd2['CDELT1']

# Both headers are the same so one needs only 2 variables
cdelt = cdelt1
crval = crval1

# x axis of the spectrums 
x = 10**(np.linspace(0,7999,8000)*cdelt + crval)
#x = np.arange(cdelt,crval,8000)

# Renormalization of the spectrums
k_spectrum = k_spectrum/np.max(k_spectrum)
galaxy_spectrum = galaxy_spectrum/np.max(galaxy_spectrum)

# curve fit on the renormalized spectrums
p = np.array([1])
popt, pcov = curve_fit(convolution_gauss, k_spectrum, galaxy_spectrum, p)  

print('Radial velocity dispersion is :', popt[0])
print('\n') 

F = convolution_gauss(k_spectrum,popt)

result = galaxy_spectrum - F

# PLOT

# plot of the spectrum
plt.figure()
plt.plot(x, k_spectrum, color = 'red', label='Star spectrum')
plt.plot(x, galaxy_spectrum, color='blue', label='Galaxy spectrum')
plt.axvline(x=3933.44,color='b', linestyle='--',label='Ca, K band')
plt.axvline(x=3969.17,color='g', linestyle='--',label='Ca, H band')
plt.axvline(x=4303.36,color='c', linestyle='--',label='CH molecules, G band')
plt.axvline(x=4861.32,color='m', linestyle='--',label='H_B')
plt.axvline(x=5174.00,color='y', linestyle='--',label='Mg band')
plt.axvline(x=5892.36,color='k', linestyle='--',label='Na doublet')
plt.axvline(x=6562.80,color='orange', linestyle='--',label='H_a')
plt.legend(loc='lower right',ncol=2)
plt.show()
# Add title and axis names
plt.xlabel('Wavelength $\lambda$')
plt.ylabel('Intensity')

# plot of the spectrum
plt.figure()
plt.plot(x, galaxy_spectrum, color='blue', label='b')
plt.axvline(x=3933.44,color='r', linestyle='--',label='Ca, K band')
plt.axvline(x=3969.17,color='g', linestyle='--',label='Ca, H band')
plt.axvline(x=4303.36,color='c', linestyle='--',label='CH molecules, G band')
plt.axvline(x=4861.32,color='m', linestyle='--',label='H_B')
plt.axvline(x=5174.00,color='y', linestyle='--',label='Mg band')
plt.axvline(x=5892.36,color='k', linestyle='--',label='Na doublet')
plt.axvline(x=6562.80,color='orange', linestyle='--',label='H_a')
plt.legend(loc='lower right',ncol=3)
plt.show()
# Add title and axis names
plt.title('Galaxy spectrum')
plt.xlabel('Wavelength $\lambda$')
plt.ylabel('')

# plot of the spectrum
plt.figure()
plt.plot(x, F, color='green', label='c')
plt.show()
# Add title and axis names
plt.title('Observed galaxy spectrum')
plt.xlabel('Wavelength $\lambda$')
plt.ylabel('')

plt.figure()
plt.plot(x, F, x, galaxy_spectrum, label='a')
plt.show()
# Add title and axis names
plt.title('F - G')
plt.xlabel('Wavelength $\lambda$')
plt.ylabel('G - F $\cdot S$')
#%%
# plot of the spectrum
plt.figure()
plt.plot(x, np.abs(result), color = 'red')
plt.show()
plt.legend()
plt.grid()
# Add title and axis names
plt.xlabel('Wavelength $\lambda$')
plt.ylabel('|G - F $\cdot S$|')

rho2 = ((popt[0]/3)**2)/(2*np.pi*G)
Mass = 4 * np.pi * rho2 * 0.8 * np.log(10)*c*cdelt*10**-3

sampling = np.log(10)*c*cdelt*10**-3

Mass = 6*(popt[0]*sampling*3.24078e-17)**2 *0.8/G 

print('The mass of the galaxy is :', "%e"%Mass)
print('\n') 

#%%

# #-------------- Part 3 --------------    
print('------ Part 3 ------')
print('\n') 

# loading of spectrums templates
file_name = "./elliptical_template.fits"
rawdata = pyfits.open(file_name) 
eltemp = rawdata[0].data
cdelt = rawdata[0].header['CDELT1']
crval = rawdata[0].header['CRVAL1']
x_ellipt = 10**(np.linspace(0,len(eltemp)-1,len(eltemp))*cdelt + crval)
# plot of the spectrum
# plt.figure()
# plt.title('elliptical')
# plt.plot(x_ellipt,eltemp, color = 'red', label='a')
# plt.show()


file_name = "./s0_template.fits"
rawdata = pyfits.open(file_name) 
s0_temp = rawdata[0].data 
cdelt = rawdata[0].header['CDELT1']
crval = rawdata[0].header['CRVAL1']
x_s0 = 10**(np.linspace(0,len(s0_temp)-1,len(s0_temp))*cdelt + crval)
# plot of the spectrum
# plt.figure()
# plt.title('s0')
# plt.plot(x_s0,s0_temp, color = 'red', label='a')
# plt.show()

file_name = "./sa_template.fits"
rawdata = pyfits.open(file_name) 
sa_temp = rawdata[0].data 
cdelt = rawdata[0].header['CDELT1']
crval = rawdata[0].header['CRVAL1']
x_sa = 10**(np.linspace(0,len(sa_temp)-1,len(sa_temp))*cdelt + crval)
# plot of the spectrum
# plt.figure()
# plt.title('sa')
# plt.plot(x_sa,sa_temp, color = 'red', label='a')
# plt.show()

file_name = "./sb_template.fits"
rawdata = pyfits.open(file_name) 
sb_temp = rawdata[0].data  
cdelt_sb = rawdata[0].header['CDELT1']
crval_sb = rawdata[0].header['CRVAL1']
x_sb = 10**(np.linspace(0,len(sb_temp)-1,len(sb_temp))*cdelt_sb + crval_sb) 
# plot of the spectrum
# plt.figure()
# plt.title('sb')
# plt.plot(x_sb,sb_temp, color = 'red', label='a')
# plt.show()

file_name = "./galaxy01.fits"
rawdata = pyfits.open(file_name) 
galax1 = rawdata[0].data  
cdelt1 = rawdata[0].header['CDELT1']
crval1 = rawdata[0].header['CRVAL1']
xg1 = 10**(np.linspace(0,len(galax1)-1,len(galax1))*cdelt + crval) 
# plot of the spectrum
# plt.figure()
# plt.plot(xg1,galax1)
# plt.show()


z = np.zeros(20) # redshift
# # delta = 0
# # chi = 100
for i in range(1,21):
    
    if i == 20 or 16 or 12 or 8 or 4:
        
        file_name = "./galaxy0" + str(i) + ".fits"
        
        data = pyfits.open(file_name) 
        spec = data[0].data
        crval = data[0].header['CRVAL1']
        cdelt = data[0].header['CDELT1']
        x = 10**(np.linspace(0,len(spec)-1,len(spec))*cdelt + crval)    

        
        ind_NA = round((np.log10(5892.36)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        ind_NA = round((np.log10(5174.0)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = z[i-1] + (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        ind_NA = round((np.log10(4303.36)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = z[i-1] + (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        z[i-1] = z[i-1]/3
        
    if i == 18 or 14 or 10 or 6 or 2:
        
        file_name = "./galaxy0" + str(i) + ".fits"
        
        data = pyfits.open(file_name) 
        spec = data[0].data
        crval = data[0].header['CRVAL1']
        cdelt = data[0].header['CDELT1']
        x = 10**(np.linspace(0,len(spec)-1,len(spec))*cdelt + crval)    
      
        
        ind_NA = round((np.log10(5892.36)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        ind_NA = round((np.log10(5174.0)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = z[i-1] + (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        ind_NA = round((np.log10(4861.32)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = z[i-1] + (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        z[i-1] = z[i-1]/3
        
    if i == 17 or 13 or 9 or 5 or 1:
        
        file_name = "./galaxy0" + str(i) + ".fits"
        
        data = pyfits.open(file_name) 
        spec = data[0].data
        crval = data[0].header['CRVAL1']
        cdelt = data[0].header['CDELT1']
        x = 10**(np.linspace(0,len(spec)-1,len(spec))*cdelt + crval)    
    
        
        ind_NA = round((np.log10(5892.36)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        ind_NA = round((np.log10(5174.0)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = z[i-1] + (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        ind_NA = round((np.log10(4303.36)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = z[i-1] + (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        z[i-1] = z[i-1]/3
        
    if i == 19 or 15 or 11 or 7 or 3:
        file_name = "./galaxy0" + str(i) + ".fits"  
        
        data = pyfits.open(file_name) 
        spec = data[0].data
        crval = data[0].header['CRVAL1']
        cdelt = data[0].header['CDELT1']
        x = 10**(np.linspace(0,len(spec)-1,len(spec))*cdelt + crval)    
      
        
        ind_NA = round((np.log10(5892.36)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        ind_NA = round((np.log10(5174.0)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = z[i-1] + (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        ind_NA = round((np.log10(4303.36)-crval)/cdelt)  # theoretical index of NA doublet in template  
        ind_min = np.argmin(spec[ind_NA-15:ind_NA+15])   # index of the NA doublet in spectrum of galaxies
        z[i-1] = z[i-1] + (10**((ind_min+ind_NA-15)*cdelt+crval)-10**(ind_NA*cdelt_sb+crval_sb))/10**(ind_NA*cdelt_sb+crval_sb)   # redshift computation
        
        z[i-1] = z[i-1]/3


v_rad = c * (((z+1)**2-1)/((z+1)**2+1))*10**-3  # radial velocities in km/s

sigma_r = np.sqrt(1/19*sum((v_rad-np.mean(v_rad))**2))    # velocity dispersion

# M = 6*(sigma_r*3.24078e-17)**2 *620/G   # mass of the cluster 
M = 2 * ((0.8*sigma_r**2)/(G))/(1.989*10**30)


print('The mass of the cluster is :',"%e"%M)
print('\n') 
print('The velocity dispersion is :',"%e"%sigma_r)
print('\n') 









































#         M = np.zeros((len(spec), 2))
#         Mtampon = np.zeros((len(spec), 2))
#         M[:,0] = x
#         M[:,1] = eltemp    
#         values = np.ones(len(spec))
#         iter = np.ones(5)
#         print(i)
#         for k in range(0.00001,0001,10): 
#             print(k)
#             A = values * (k/10)
#             M[:,0] = M[:,0] - A
#             for j in range(10,1730,1721):               
#                 chitampon = ((spec[j] - M[j,1])**2)*np.sqrt(spec[j])
#                 if chitampon < chi:
#                     chi = chitampon
#                     delta = k                    
#         print('delta pour la ', i, 'galaxie est',delta)
              



















    

