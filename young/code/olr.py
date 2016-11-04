from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model

multislice_dic = loadmat('../data/Pt110-multislice-v0.mat')
convolution = np.load('../data/Pt110-convolution-v0.npy')
multislice = multislice_dic['ImgG']

### Simple Linear Regression - reduce all pixels to one vector
multislice1D = multislice.reshape(-1,1)
convolution1D = convolution.reshape(-1,1)
bias = np.ones(convolution1D.shape)

X = np.concatenate((bias, convolution1D), axis=1)

regr = linear_model.LinearRegression()
regr.fit(X, multislice1D)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.8f"
      % np.mean((regr.predict(X) - multislice1D) ** 2))

plt.ion()
plt.figure(1)
plt.scatter(convolution1D, multislice1D, linewidth=3)
plt.plot(convolution1D, regr.predict(X), linewidth=3)
plt.title('Convolution vs. Multislice - Simple Linear Regression')
plt.xlabel('Convolution')
plt.ylabel('Multislice')
plt.savefig('slr.png', dpi=300)
plt.show()

plt.ion()
plt.figure(2)
plt.scatter(regr.predict(X), multislice1D, linewidth=3)
plt.plot([0,1,2,3])
plt.title('Order 1 Multiple Linear Regression - 1 pixel mapping')
plt.xlabel('Predicted')
plt.ylabel('Multislice')
plt.xlim(-0.05, 0.3)
plt.ylim(-0.05, 0.3)
plt.savefig('olr_order1_1pixel.png', dpi=300)
plt.show()
