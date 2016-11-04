from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model

mult_dic = loadmat('../data/Pt110-multislice-v0.mat')
conv = np.load('../data/Pt110-convolution-v0.npy')
mult = mult_dic['ImgG']

### Simple Linear Regression - reduce all pixels to one vector
mult1D = mult.reshape(-1,1)
conv1D = conv.reshape(-1,1)
bias = np.ones(conv1D.shape) # for intercept

X = np.concatenate((bias, conv1D), axis=1)
regr = linear_model.LinearRegression()
regr.fit(X, mult1D)

### Multiple Linear Regression - one pixel mapping
X2 = np.concatenate((bias, conv1D, conv1D**2), axis=1)
regr2 = linear_model.LinearRegression()
regr2.fit(X2, mult1D)

# Plot the one pixel mapping (2nd order)
plt.ion()
plt.figure(1)
plt.scatter(regr2.predict(X2), mult1D, linewidth=3)
plt.plot([0,1,2,3])
plt.title('Order 2 Multiple Linear Regression - 1 pixel mapping')
plt.xlabel('Predicted')
plt.ylabel('Multislice')
plt.xlim(-0.05, 0.3)
plt.ylim(-0.05, 0.3)
plt.savefig('olr_order2_1pixel.png', dpi=300)
plt.show()

### Extract 8-neighbors
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


### Regression using 9-pixels (first order)
conv9 = neighbors.reshape(-1,9)

X3 = np.concatenate((bias, conv9), axis=1)
regr3 = linear_model.LinearRegression()
regr3.fit(X3, mult1D)

# Plot the 9 pixel mapping (1st order)
plt.figure(2)
plt.scatter(regr3.predict(X3), mult1D, linewidth=3)
plt.plot([0,1,2,3])
plt.title('Order 1 Multiple Linear Regression - 9 pixel mapping')
plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.xlim(-0.05, 0.3)
plt.ylim(-0.05, 0.3)
plt.savefig('olr_order1_9pixel.png', dpi=300)
plt.show() 

### Regression using 9-pixels (second order)
X4 = np.concatenate((bias, conv9, conv9**2), axis=1)
regr4 = linear_model.LinearRegression()
regr4.fit(X4, mult1D)

# Plot the 9 pixel mapping (1st order)
plt.figure(3)
plt.scatter(regr4.predict(X4), mult1D, linewidth=3)
plt.plot([0,1,2,3])
plt.title('Order 2 Multiple Linear Regression - 9 pixel mapping')
plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.xlim(-0.05, 0.3)
plt.ylim(-0.05, 0.3)
plt.savefig('olr_order2_9pixel.png', dpi=300)
plt.show() 

### Use gradient information
grady = np.gradient(conv, axis=0).reshape(-1,1)
gradx = np.gradient(conv, axis=1).reshape(-1,1)
gradmag = (gradx**2 + grady**2)**0.5

X5 = np.concatenate((bias, conv9, gradx, grady, gradmag), axis=1)
regr5 = linear_model.LinearRegression()
regr5.fit(X5, mult1D)

# Plot the 9 pixel mapping (1st order)
plt.figure(4)
plt.scatter(regr5.predict(X5), mult1D, linewidth=3)
plt.plot([0,1,2,3])
plt.title('Order 1 Multiple Linear Regression - 9 pixel mapping/gradient')
plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.xlim(-0.05, 0.3)
plt.ylim(-0.05, 0.3)
plt.savefig('olr_order1_9pixel_grad.png', dpi=300)
plt.show() 

X6 = np.concatenate((bias, conv9, conv9**2, gradx, grady, gradmag), axis=1)
regr6 = linear_model.LinearRegression()
regr6.fit(X6, mult1D)

# Plot the 9 pixel mapping (1st order)
plt.figure(5)
plt.scatter(regr6.predict(X6), mult1D, linewidth=3)
plt.plot([0,1,2,3])
plt.title('Order 2 Multiple Linear Regression - 9 pixel mapping/gradient')
plt.xlabel('Predicted')
plt.ylabel('Observed')
plt.xlim(-0.05, 0.3)
plt.ylim(-0.05, 0.3)
plt.savefig('olr_order2_9pixel_grad.png', dpi=300)
plt.show() 

