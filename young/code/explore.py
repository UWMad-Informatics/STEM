from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import linear_model

mult_dic = loadmat('../data/Pt110-multislice-v0.mat')
conv = np.load('../data/Pt110-convolution-v0.npy')
mult = mult_dic['ImgG']

plt.ion()
for i in np.arange(conv.shape[2]):
    fig = plt.figure(i+1)
    fig.suptitle('Level ' + str(i))
    ax1 = fig.add_subplot(131)
    # Normalize Images
    conv_norm = (conv[:,:,i] - conv[:,:,i].mean())/conv[:,:,i].std()
    mult_norm = (mult[:,:,i] - mult[:,:,i].mean())/mult[:,:,i].std()

    # Convolution
    ax1.pcolormesh(conv_norm, cmap=cm.viridis)
    ax1.set_xlim(0, conv.shape[1])
    ax1.set_ylim(0, conv.shape[0])
    ax1.set_title('Conv')

    # Multislice
    ax2 = fig.add_subplot(132)
    ax2.pcolormesh(mult_norm, cmap=cm.viridis)
    ax2.set_xlim(0, mult.shape[1])
    ax2.set_ylim(0, mult.shape[0])
    ax2.set_title('Mult')

    # Difference
    ax3 = fig.add_subplot(133)
    ax3.pcolormesh(conv_norm - mult_norm, cmap=cm.viridis)
    ax3.set_xlim(0, mult.shape[1])
    ax3.set_ylim(0, mult.shape[0])
    ax3.set_title('Conv-Mult')
    plt.savefig('layer'+str(i)+'.png', dpi=300)
    fig.show()
    


