# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:26:55 2016

@author: aidan
"""

from scipy.io import loadmat
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

datafile1 = loadmat('Pt110-multislice-v0.mat')
#print ("The object is of type:", type(datafile1))
#print ("The keys of each value in the dictionary are...")
#for key in datafile1:
 #   print(key)

import numpy as np
datafile2 = np.load('Pt110-convolution-v0.npy')
#print("The object is of type", type(datafile2))

import matplotlib.pyplot as plt
import matplotlib.cm as cm

multislice_data = datafile1['ImgG']
#print('The shape of the dataset is:', multislice_data.shape)
#image = multislice_data[:,:,-1]
#fig, ax = plt.subplots(figsize=(4,4))
#fig.colorbar(ax.pcolormesh(image, cmap = cm.viridis))
#ax.set_aspect('equal', 'box')
#ax.set_xlim(0, image.shape[1])
#ax.set_ylim(0, image.shape[0])
#plt.title('Pt110 - 10 layers - multislice', size = 11)
#plt.savefig('Pt110-multislice-10-layers.png', dpi = 300)
#plt.show()

#print("The intensity at point (0,0) is", image[0][0])
#print("The intensity at point (2,1) is", image[2][1])

convolution_data = datafile2
#image = convolution_data[:,:,-1]
#fig, ax = plt.subplots(figsize=(4,4))
#fig.colorbar(ax.pcolormesh(image, cmap = cm.viridis))
#ax.set_aspect('equal', 'box')
#ax.set_xlim(0, image.shape[1])
#ax.set_ylim(0, image.shape[0])
#plt.title('Pt110 - 10 layers - convolution', size = 11)
#plt.savefig('Pt110-convolution-10-layers.png', dpi = 300)
#plt.show()

multislice_vector = multislice_data[:,:,-1].reshape(-1,1)
convolution_vector = convolution_data[:,:,-1].reshape(-1,1)

regression = linear_model.LinearRegression()
regression.fit(convolution_vector, multislice_vector)

plt.scatter(convolution_vector, multislice_vector, color = 'black')
plt.title('Convolution vs Multislice');
plt.plot(convolution_vector, regression.predict(convolution_vector), color = 'blue')

polyfit = make_pipeline(PolynomialFeatures(10), Ridge())
polyfit.fit(convolution_vector, multislice_vector)
plt.plot(convolution_vector, polyfit.predict(convolution_vector), color = 'red')

plt.show()
