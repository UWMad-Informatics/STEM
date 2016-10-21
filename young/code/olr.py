from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model

multislice_dic = loadmat('Pt110-multislice-v0.mat')
convolution = np.load('Pt110-convolution-v0.npy')
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
plt.scatter(convolution1D, multislice1D, linewidth=3)
plt.plot(convolution1D, regr.predict(X), linewidth=3)
plt.xticks(())
plt.yticks(())
plt.savefig('slr.png', dpi=300)
plt.show() ## need axes and titles

