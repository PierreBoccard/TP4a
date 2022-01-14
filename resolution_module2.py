#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 11:57:44 2021

@author: PierreBoccard
"""

from scipy.ndimage import gaussian_filter
import os
import numpy as np  # arrays manipulation
import math
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
import matplotlib.pyplot as plt  # plots
from PIL import Image  # images manipulation


# definition of the 2D Gaussian distribution
def gaussian2D(x, y, mux, muy, sig, A, c):  
    return np.exp( - ((x - mux)**2 + (y - muy)**2) / (2 * sig**2)) * A + c

# function called by curve_fit, uses a 2D array to store x and y axis
def gaussian(M, a, b , c, d, e):
    x, y = M 
    return gaussian2D(x, y, a, b, c, d, e)


# function that returns the standard deviation of an image; given centers of selected stars and a number of pixels, 
# where the fitting is done (we fit a 2D gaussian)
# plotR/V/B ask if we want to have a plot of each stars and the fit in the corresponding filter
def getting_deviation(star_center, N, image_storedR, image_storedV, image_storedB, plotR = True, plotV = True, plotB = True):
    Nt = int((N-1)/2)

    sigmaR = np.zeros(len(star_center))
    sigmaV = np.zeros(len(star_center))
    sigmaB = np.zeros(len(star_center))
    errB = np.zeros(len(star_center))
    errV = np.zeros(len(star_center))
    errR = np.zeros(len(star_center))
    
    print("Parameters are : mux, muy, sigma, A, c")
    print('\n')

    for i in range(len(star_center)):
 
    
        center = star_center[i]
        # x and y from DS9 are inverted
        up_x = center[0] + Nt  #upper boundary of x
        down_x = center[0] - Nt  #lower boundary of x
        up_y = center[1] + Nt   #upper boundary of y
        down_y = center[1] - Nt   #lower boundary of y
    

        contourR = image_storedR[down_x-1:up_x,down_y-1:up_y]  #reduces the image to a star only 
        contourV = image_storedV[down_x-1:up_x,down_y-1:up_y]
        contourB = image_storedB[down_x-1:up_x,down_y-1:up_y]


        # set up the grid of coordinate
        x = np.linspace(down_x, up_x, N)
        y = np.linspace(down_y, up_y, N)
        xx,yy = np.meshgrid(x, y)


        # reshape of all the datas to a 1D array
        xdata = np.vstack((xx.ravel(), yy.ravel()))
        ydataR = contourR.ravel()
        ydataV = contourV.ravel()
        ydataB = contourB.ravel()
        x, y = xdata 
        

        p0 = np.array([center[0], center[1], 1, 350, 2000]) #initial guess for the parameters

        poptR, pcovR = curve_fit(gaussian, xdata, ydataR, p0) 
        poptV, pcovV = curve_fit(gaussian, xdata, ydataV, p0)
        poptB, pcovB = curve_fit(gaussian, xdata, ydataB, p0)
        
        
        sigmaR[i] = poptR[2]
        sigmaV[i] = poptV[2]
        sigmaB[i] = poptB[2]
        errR[i] = pcovR[2,2]
        errV[i] = pcovV[2,2]
        errB[i] = pcovB[2,2]
        
  
    

        # quantites of interest
        print("Star ", i+1, ":")
        print('\n')
        print("Parameters for Red filter:",poptR)
        print('\n')
        print("Parameters for Green filter:",poptV)
        print('\n')
        print("Parameters for Blue filter:",poptB)
        print('\n')

        #plt.figure()
        #plt.imshow(dataf)
        #plt.show()
        
  
        
        if plotR == True:
            # plt.figure()
            # plt.imshow(contourR)
            # plt.show
            
            supposed_gauss = gaussian(xdata, poptR[0],poptR[1],poptR[2],poptR[3],poptR[4]) 
            dataf = supposed_gauss.reshape(N,N)    #reshape, final image
            
            # plt.figure()
            # plt.imshow(dataf)
            # plt.show()
            
        if plotV == True:
            # plt.figure()
            # plt.imshow(contourV)
            # plt.show
            
            supposed_gauss = gaussian(xdata, poptV[0],poptV[1],poptV[2],poptV[3],poptV[4]) 
            dataf = supposed_gauss.reshape(N,N)    #reshape, final image
            
            # plt.figure()
            # plt.imshow(dataf)
            # plt.show()
            
        if plotB == True:
            # plt.figure()
            # plt.imshow(contourB)
            # plt.show
            
            supposed_gauss = gaussian(xdata, poptB[0],poptB[1],poptB[2],poptB[3],poptB[4]) 
            dataf = supposed_gauss.reshape(N,N)    #reshape, final image
            
            # plt.figure()
            # plt.imshow(dataf)
            # plt.show()                        
        #plt.imshow(contourV)
        #plt.imshow(contourB)
        #plt.show
            
            
    return sigmaR, sigmaV, sigmaB, errR, errV, errB
    




