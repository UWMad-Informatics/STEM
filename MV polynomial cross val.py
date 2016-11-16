# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 18:35:20 2016

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

for m in range(1,100):      
    #Data--20%
      
    #Generate random indices to remove      
    randomi = []
    randomj = []
    numremove = round(elements * .0)  
                
    for n in range(0 , numremove):
        i = random.randint(1,shape[0]-2)
        j = random.randint(1,shape[1]-2)  
        while repeatindex(i, j, randomi, randomj) == True:
            i = random.randint(1,shape[0]-2)
            j = random.randint(1,shape[1]-2)
        
        randomi.append(i)
        randomj.append(j)
                
    #Build vector of all slices excluding random indices
                
    X = []
    allX = []
    mstofit = []
    mstofitall = []
    removed = 0
    
    for k in range (0, shape[2]-1):
        for i in range (1, shape[0]-1):
            for j in range (1, shape[1]-1):
                n = 0
                out = False
                
                #all convolution data to vector
                allX.append([data[i][j][k], data[i+1][j][k], data[i][j+1][k], 
                      data[i-1][j][k], data[i][j-1][k], data[i+1][j+1][k], 
                      data[i+1][j-1][k], data[i-1][j+1][k], data[i-1][j-1][k]])
                mstofitall.append(multislice_data[i][j][k])  
                
                #vector with selected indices removed
                while(n < numremove and out == False):
                    if (randomi[n] == i and randomj[n] == j):
                        out = True
                        removed = removed + 1
                    n = n + 1
                if out == False:
                    X.append([data[i][j][k], data[i+1][j][k], data[i][j+1][k], 
                      data[i-1][j][k], data[i][j-1][k], data[i+1][j+1][k], 
                      data[i+1][j-1][k], data[i-1][j+1][k], data[i-1][j-1][k]])
                    mstofit.append(multislice_data[i][j][k])
           
    #Multivariate Polynomial Model
    
    poly = PolynomialFeatures(degree=4)
    
    #transforms data to polynomial representation necessary for desired degree
    Xtransform = poly.fit_transform(X)
    allXtransform = poly.fit_transform(allX)
    
    #fits model using data of reduced vector
    polymvreg = linear_model.LinearRegression()
    polymvreg.fit(Xtransform, mstofit)
    
    #predicts all points using model
    predicted = polymvreg.predict(allXtransform)
    
    #plots result
#    plt.scatter(predicted, mstofitall)
#    plt.plot([0,.2], [0,.2], color = 'r', linestyle = '-')
#    plt.title('Multivariate Polynomial Model')
#    plt.xlabel('Predicted')
#    plt.ylabel('Actual')
#    plt.show()
    
    #root mean square
    rms.append(sqrt(mean(square(predicted - mstofitall))))
    
avrms = np.average(rms)
stddev = np.std(rms)
elapsedtime = time.time() - starttime

print('Average multivariate polynomial RMS (100 runs): ', avrms)
print('Standard deviation of rms (100 runs): ', stddev)
print('Time Elapsed: ', elapsedtime)