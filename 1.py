#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:10:13 2021

@author: PierreBoccard
"""

#!/usr/bin/env python3

from scipy.ndimage import gaussian_filter
import os
import numpy as np  # arrays manipulation
import astropy.io.fits as pyfits  # open / write FITS files
import matplotlib.pyplot as plt  # plots
from PIL import Image  # images manipulation
import math
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import resolution_module
import resolution_module2
import resolution_modul


# In the 4 loadings the x edges are removed because of the 
# low quality of measure in the ff

# loading of the VLT images
file_nameR = "NGC613R.fits"
image_stored_rawR = pyfits.open(file_nameR) 
data_R = image_stored_rawR[0].data[:,18:-18]

file_nameV = "NGC613V.fits"
image_stored_rawV = pyfits.open(file_nameV)
data_V = image_stored_rawV[0].data[:,18:-18]

file_nameB = "NGC613B.fits"
image_stored_rawB = pyfits.open(file_nameB)
data_B = image_stored_rawB[0].data[:,18:-18]

# loading of the biases
file_name = "bias1.fits"
image_stored_raw = pyfits.open(file_name) 
bias1 = image_stored_raw[0].data[:,18:-18]

file_name = "bias2.fits"
image_stored_raw = pyfits.open(file_name) 
bias2 = image_stored_raw[0].data[:,18:-18]

file_name = "bias3.fits"
image_stored_raw = pyfits.open(file_name) 
bias3 = image_stored_raw[0].data[:,18:-18]

file_name = "bias4.fits"
image_stored_raw = pyfits.open(file_name) 
bias4 = image_stored_raw[0].data[:,18:-18]

# loading of the flatfields
file_name = "flatB.fits"
image_stored_raw = pyfits.open(file_name) 
flatfield1 = image_stored_raw[0].data[:,18:-18]

file_name = "flatV.fits"
image_stored_raw = pyfits.open(file_name) 
flatfield2 = image_stored_raw[0].data[:,18:-18]

file_name = "flatR.fits"
image_stored_raw = pyfits.open(file_name) 
flatfield3 = image_stored_raw[0].data[:,18:-18]

# # loading of the sky
# file_name = "STD1.fits"
# image_stored_raw = pyfits.open(file_name) 
# sky1 = image_stored_raw[0].data[:,18:-18]

# file_name = "STD2.fits"
# image_stored_raw = pyfits.open(file_name) 
# sky2 = image_stored_raw[0].data[:,18:-18]

# file_name = "STD3.fits"
# image_stored_raw = pyfits.open(file_name) 
# sky3 = image_stored_raw[0].data[:,18:-18]


#---------Part 1.3 --------------------------
print('------ Part 1.3 ------')
print('\n')

# loading of the full VLT images
file_nameR = "NGC613R.fits"
image_stored_rawR = pyfits.open(file_nameR) 
image_storedR = image_stored_rawR[0].data

file_nameV = "NGC613V.fits"
image_stored_rawV = pyfits.open(file_nameV)
image_storedV = image_stored_rawV[0].data

file_nameB = "NGC613B.fits"
image_stored_rawB = pyfits.open(file_nameB)
image_storedB = image_stored_rawB[0].data

# centers of the stars selected [y,x]
# star_center = [[1278,2028],[1347,546],[572,1027],[1549,1310],[1197,1982],[155,1700],[861,558],[1210,1406]]
#star_center = [[1561,289],[1792,319],[1600,890],[1326,703],[810,57],[572,1026],[1960,1913]]
star_center = [[1278,2028],[1347,546],[572,1027],[1549,1310],[1197,1982],[155,1700],[861,558],[1210,1406]]

#star_center = [[1792,319]]

  
# this function returns the standard deviation of a 2D gaussian
# sigmaR, sigmaV, sigmaB, errR, errV, errB = resolution_module2.getting_deviation(star_center, 21, image_storedR, image_storedV, image_storedB)
sigmaR, sigmaV, sigmaB, errR, errV, errB = resolution_modul.getting_deviation(star_center, 21, image_storedR, image_storedV, image_storedB)


resoR = 2.355*np.mean(sigmaR)
resoV = 2.355*np.mean(sigmaV)
resoB = 2.355*np.mean(sigmaB)

print(resoR,resoV,resoB,2.355*np.std(sigmaR)/np.sqrt(8),2.355*np.std(sigmaV)/np.sqrt(8),2.355*np.std(sigmaB)/np.sqrt(8))

#-----------Part 1.4 --------------------------
print('------ Part 1.4 ------')
print('\n')

# usefull quantities
Nx = len(bias1[0])
Ny = len(bias1)

# work on biases
best_bias = np.mean((bias1+bias2+bias3)/3)
std_bb = np.std((bias1+bias2+bias3)/3)
print("Our bias is the mean of b1 b2 b3 and has std: ", std_bb)
print('\n')

# flatfield modifier
ff1 = flatfield1 - best_bias
ff2 = flatfield2 - best_bias
ff3 = flatfield3 - best_bias

ff1 = ff1/np.mean(ff1)  #blue ff normalized
ff2 = ff2/np.mean(ff2)  #green ff normalized
ff3 = ff3/np.mean(ff3)  #red ff normalized

# data_R = np.flip(data_R,axis=0)
# data_V = np.flip(data_V,axis=0)
# data_B = np.flip(data_B,axis=0)

# work on the sky, we take the mean of the flux in some areas without stars
imR = (data_R - best_bias)/ff3
imV = (data_V - best_bias)/ff2
imB = (data_B - best_bias)/ff1

# R1 = np.mean(imR[1030:1230,40:260])
# R2 = np.mean(imR[20:280,50:250])
# R3 = np.mean(imR[1700:1950,1200:1600])

# V1 = np.mean(imV[1030:1230,40:260])
# V2 = np.mean(imV[20:280,50:250])
# V3 = np.mean(imV[1700:1950,1200:1600])


# B1 = np.mean(imB[1030:1230,40:260])
# B2 = np.mean(imB[20:280,50:250])
# B3 = np.mean(imB[1700:1950,1200:1600])

R1 = np.mean(imR[1568:1736,1593:1888])
R2 = np.mean(imR[335:456,149:320])
R3 = np.mean(imR[303:419,1328:1563])
#R4 = np.mean(imR[1580:1717,48:173])

V1 = np.mean(imV[1568:1736,1593:1888])
V2 = np.mean(imV[335:456,149:320])
V3 = np.mean(imV[303:419,1328:1563])
#V4 = np.mean(imV[1580:1717,48:173])

B1 = np.mean(imB[1568:1736,1593:1888])
B2 = np.mean(imB[335:456,149:320])
B3 = np.mean(imB[303:419,1328:1563])


skylvlR = np.mean([R1,R2,R3])
skylvlV = np.mean([V1,V2,V3])
skylvlB = np.mean([B1,B2,B3])

# work on datas, returns the data of interest
image_R = np.true_divide((data_R - best_bias),ff3) - skylvlR
image_V = np.true_divide((data_V - best_bias),ff2) - skylvlV
image_B = np.true_divide((data_B - best_bias),ff1) - skylvlB

# applies the convolution
final_image_R = gaussian_filter(image_R, (resoB**2-resoR**2)**0.5)
final_image_V = gaussian_filter(image_V, (resoB**2-resoV**2)**0.5)
final_image_B = image_B

R1 = np.mean(image_R[1871:1958,1415:1625])
R2 = np.mean(image_R[254:350,1542:1738])
R3 = np.mean(image_R[146:257,674:1000])

V1 = np.mean(image_V[1871:1958,1415:1625])
V2 = np.mean(image_V[254:350,1542:1738])
V3 = np.mean(image_V[146:257,674:1000])

B1 = np.mean(image_B[1871:1958,1415:1625])
B2 = np.mean(image_B[254:350,1542:1738])
B3 = np.mean(image_B[146:257,674:1000])

# plt.figure()
# plt.imshow(final_image_R,norm=LogNorm(1000, 6300), cmap='gray')
# plt.show

# plt.figure()
# plt.imshow(final_image_V,norm=LogNorm(1000, 6300), cmap='gray')
# plt.show()

# plt.figure()
# plt.imshow(final_image_R,norm=LogNorm(1000, 6300), cmap='Blues')
# plt.show

# plt.figure()
# plt.imshow(final_image_B,norm=LogNorm(1000, 6300), cmap='Blues')
# plt.show()

k = [0.001 if i[j]<=0 else i[j] for i in final_image_R for j in range(len(i))]
l = [0.001 if i[j]<=0 else i[j] for i in final_image_B for j in range(len(i))]
k = np.array(k)
l = np.array(l)

RB_ratio = -2.5*np.log10(k/l)
RB_ratio = RB_ratio.reshape(Ny,Nx)


# creation of output files
# hdu = pyfits.PrimaryHDU(RB_ratio)
# hdu_list = pyfits.HDUList([hdu])
# hdu_list.writeto(os.path.join("/Users/PierreBoccard/Documents/EPFL/MA1/Labo/Exercice1", "color_map.fits"))

# hdu = pyfits.PrimaryHDU(final_image_R)
# hdu_list = pyfits.HDUList([hdu])
# hdu_list.writeto(os.path.join("/Users/PierreBoccard/Documents/EPFL/MA1/Labo/Exercice1", "image_final_R.fits"))

# hdu = pyfits.PrimaryHDU(final_image_V)
# hdu_list = pyfits.HDUList([hdu])
# hdu_list.writeto(os.path.join("/Users/PierreBoccard/Documents/EPFL/MA1/Labo/Exercice1", "image_final_V.fits"))

# hdu = pyfits.PrimaryHDU(final_image_B)
# hdu_list = pyfits.HDUList([hdu])
# hdu_list.writeto(os.path.join("/Users/PierreBoccard/Documents/EPFL/MA1/Labo/Exercice1", "image_final_B.fits"))

#------------------ Part 1.5 ----------------------------------
print('------ Part 1.5 ------')
print('\n')

# loading of the full HST images
file_nameR = "HSTred_big.fits"
image_stored_rawR = pyfits.open(file_nameR) 
HST_imR = image_stored_rawR[0].data

file_nameR = "HSTgreen_big.fits"
image_stored_rawR = pyfits.open(file_nameR) 
HST_imV = image_stored_rawR[0].data

file_nameR = "HSTblue_big.fits"
image_stored_rawR = pyfits.open(file_nameR) 
HST_imB = image_stored_rawR[0].data


# centers of the selected stars 
star_center = [[196,1116],[157,1115],[587,1348],[709,488],[453,658],[852,440],[1075,993],[157,1115],[587,1348],[946,1170],[1243,1497],[842,833]]
#star_center = [[196,1116],[157,1115],[587,1348],[709,488],[453,658],[852,440],[1075,993],[157,1115],[587,1348],[946,1170],[1243,1497],[842,833]]


sigmaR, sigmaV, sigmaB, errR, errV, errB = resolution_module2.getting_deviation(star_center, 15, HST_imR, HST_imV, HST_imB, plotR = False, plotV = False, plotB = False)

resoR_HST = 2.355*np.mean(sigmaR)
resoV_HST = 2.355*np.mean(sigmaV)
resoB_HST = 2.355*np.mean(sigmaB) 

HST_B = gaussian_filter(HST_imB, (resoV_HST**2-resoB_HST**2)**0.5)
HST_R = gaussian_filter(HST_imR, (resoV_HST**2-resoR_HST**2)**0.5)
HST_V = HST_imV

a = HST_B


x = [0.001 if i[j]<=0 else i[j] for i in HST_R for j in range(len(i))]
y = [0.001 if i[j]<=0 else i[j] for i in HST_B for j in range(len(i))]
x = np.array(x)
y = np.array(y)

RB_ratio_HST = -2.5*np.log10(x/y)
RB_ratio_HST = RB_ratio_HST.reshape(len(HST_B),len(HST_B[0]))

# creation of output files
# hdu = pyfits.PrimaryHDU(RB_ratio_HST)
# hdu.writeto('color_map_HST.fits')

# hdu = pyfits.PrimaryHDU(HST_R)
# hdu.writeto('final_HST_R.fits')

# hdu = pyfits.PrimaryHDU(HST_V)
# hdu.writeto('final_HST_V.fits')

# hdu = pyfits.PrimaryHDU(HST_B)
# hdu.writeto('final_HST_B.fits')
