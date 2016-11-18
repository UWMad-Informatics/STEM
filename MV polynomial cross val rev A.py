# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:35:20 2016
This file cross validates the nine surrounding point polynomial model by
removing each slice in sequence from the data set and then predicting it back.
@author: aidan
"""

from scipy.io import loadmat
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from numpy import square, mean, sqrt
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

def find(lst, n):
    result = []
    elist = enum(lst)
    for i, x in elist:
        if x == n:
            result.append(i)
    return result
    
def repeatindex(i, j, randi, randj):
        containsj = False
        containsi = i in randi
        
        if containsi == True:
            indexi = find(randi, i)
            for m in indexi:
                if randj[m] == j:
                    containsj = True
        
        return containsj
               
def enum(lst):
    elist = []
    i = 0
    for el in lst:
        elist.append([i, el])
        i = i + 1
    return elist

datafile1 = loadmat('Pt110-multislice-v0.mat')
datafile2 = np.load('Pt110-convolution-v0.npy')

multislice_data = datafile1['ImgG']
convolution_data = datafile2

multislice_vector = multislice_data[:,:,-1].reshape(-1,1)
convolution_vector = convolution_data[:,:,-1].reshape(-1,1)

shape = convolution_data.shape
data = convolution_data
elements = (shape[0]-3) * (shape[1]-3)

starttime = time.time()
rms = []
for slicetoremove in range(0, shape[2]):
            
    #Build vector of all slices excluding 1 slice
                
    X_train = []
    X_slice = []
    ms_train = []
    ms_slice = []
    
    for k in range (0, shape[2]):
        for i in range (1, shape[0]-1):
            for j in range (1, shape[1]-1):
                if k != slicetoremove:
                    n = 0
                    out = False
                    
                    #convolution data to vector excluding 1 slice
                    X_train.append([data[i][j][k], data[i+1][j][k], data[i][j+1][k], 
                          data[i-1][j][k], data[i][j-1][k], data[i+1][j+1][k], 
                          data[i+1][j-1][k], data[i-1][j+1][k], data[i-1][j-1][k]])
                    ms_train.append(multislice_data[i][j][k])
                    
                if k == slicetoremove:
                    X_slice.append([data[i][j][k], data[i+1][j][k], data[i][j+1][k], 
                          data[i-1][j][k], data[i][j-1][k], data[i+1][j+1][k], 
                          data[i+1][j-1][k], data[i-1][j+1][k], data[i-1][j-1][k]])
                    ms_slice.append(multislice_data[i][j][k])
    
    #Multivariate Polynomial Model
    
    poly = PolynomialFeatures(degree=4)
    
    #transforms data to polynomial representation necessary for desired degree
    X_train_transform = poly.fit_transform(X_train)
    X_slicetransform = poly.fit_transform(X_slice)
    
    #fits model using data of reduced vector
    polymvreg = linear_model.LinearRegression()
    polymvreg.fit(X_train_transform, ms_train)
    
    #predicts all points using model
    predicted = polymvreg.predict(X_slicetransform)
    
    #plots result
    plt.scatter(predicted, ms_slice)
    plt.plot([0,.2], [0,.2], color = 'r', linestyle = '-')
    plt.title('Multivariate Polynomial Model')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    #creates difference image map
    
    #rearrange vector into array
    predictedarray = np.reshape(predicted, (shape[0]-2, shape[1]-2))
    actualarray = np.reshape(ms_slice, (shape[0]-2, shape[1]-2))
    
    #obtain difference array
    image = predictedarray - actualarray
    
    #plots differences
    fig, ax = plt.subplots(figsize=(4,4))
    fig.colorbar(ax.pcolormesh(image, cmap = cm.viridis))
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(0, image.shape[0])
    plt.title('Differences between Predicted and Actual', size = 11)
    plt.show()
    
    print('Slice ', slicetoremove)
    
    #root mean square
    rms = sqrt(mean(square(predicted - ms_slice)))
    print('RMS: ', rms)

elapsedtime = time.time() - starttime