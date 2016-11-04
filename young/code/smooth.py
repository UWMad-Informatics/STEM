from scipy.io import loadmat
from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model

mult_dic = loadmat('../data/Pt110-multislice-v0.mat')
conv = np.load('../data/Pt110-convolution-v0.npy')
mult = mult_dic['ImgG']

conv_smooth = sg.convolve(conv[:,:,1], 1/9 * np.ones((3,3)), "same")
plt.figure(1)
plt.imshow(conv_smooth-mult[:,:,1]); plt.show()
plt.figure(2)
plt.imshow(conv[:,:,1]-mult[:,:,1]); plt.show()
