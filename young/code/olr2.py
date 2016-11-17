from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model
import time
### Data cleaning
mult_dic = loadmat('../data/Pt110-multislice-v0.mat')
conv = np.load('../data/Pt110-convolution-v0.npy')
mult = mult_dic['ImgG']

# reduce all pixels to 1D
mult1D = mult.reshape(-1,1)
conv1D = conv.reshape(-1,1)
bias = np.ones(conv1D.shape) # for intercept

# Extract 8-neighbors
t0 = time.time()
for i in range(100):
    neighbors = -1*np.ones((*conv.shape,9))
    neighbors[1:,1:,:,0] = conv[:-1,:-1,:]
    neighbors[1:,:,:,1] = conv[:-1,:,:]
    neighbors[1:,:-1,:,2] = conv[:-1,1:,:]
    neighbors[:,1:,:,3] = conv[:,:-1,:]
    neighbors[:,:,:,4] = conv[:,:,:]
    neighbors[:,:-1,:,5] = conv[:,1:,:]
    neighbors[:-1,1:,:,6] = conv[1:,:-1,:]
    neighbors[:-1,:,:,7] = conv[1:,:,:]
    neighbors[:-1,:-1,:,8] = conv[1:,1:,:]
    neighbors = np.where(neighbors == -1, conv[...,None], neighbors)
    
t1 = time.time()
total = t1-t0
print('total =' + str(total))

# 9-pixels (first order)
conv9 = neighbors.reshape(-1,9)

# Gradient information

grady = np.gradient(conv, axis=0).reshape(-1,1)
gradx = np.gradient(conv, axis=1).reshape(-1,1)
gradmag = (gradx**2 + grady**2)**0.5


### Models
# 1 pixel linear
X = np.concatenate((bias, conv1D), axis=1)
regr = linear_model.LinearRegression()
regr.fit(X, mult1D)

# 1 pixel quadratic
X2 = np.concatenate((bias, conv1D, conv1D**2), axis=1)
regr2 = linear_model.LinearRegression()
regr2.fit(X2, mult1D)

# 9 pixel linear
X3 = np.concatenate((bias, conv9), axis=1)
regr3 = linear_model.LinearRegression()
regr3.fit(X3, mult1D)

# 9 pixel quadratic
X4 = np.concatenate((bias, conv9, conv9**2), axis=1)
regr4 = linear_model.LinearRegression()
regr4.fit(X4, mult1D)

# 9 pixel linear with gradient
X5 = np.concatenate((bias, conv9, gradx, grady, gradmag), axis=1)
regr5 = linear_model.LinearRegression()
regr5.fit(X5, mult1D)

# 9 pixel quadratic with gradient
X6 = np.concatenate((bias, conv9, conv9**2, gradx, grady, gradmag), axis=1)
regr6 = linear_model.LinearRegression()
regr6.fit(X6, mult1D)

## plot
from sklearn import linear_model as lm
import matplotlib.pyplot as plt

for model in models:
    trainX = model[0][0]
    trainY = model[0][1]
    testX = model[0][0]
    testY = model[0][1]
    regr = lm.LinearRegression().fit(trainX, trainY)
    mse = np.mean((regr.predict(testX) - testY) ** 2)
    rsq = regr.score(testX, testY)
    plt.figure()
    plt.scatter(regr.predict(testX), mult, s=1)
    plt.plot([0,1,2,3])
    plt.text(0.15, 0.02, 'MSE = ' + str(round(mse,5)))
    plt.text(0.15, 0, r'R.sq =' + str(round(rsq, 3)))
    plt.title(model[1])
    plt.xlabel('Predicted')
    plt.ylabel('Observed - Multislice')
    plt.xlim(-0.05, 0.3)
    plt.ylim(-0.05, 0.2)
    plt.show()

