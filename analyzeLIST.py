#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:39:31 2017

@author: aidan
"""

from numpy import square, mean, sqrt
import numpy as np
from fractions import Fraction
from functools import reduce
from operator import mul  
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import math

#takes an error type and returns that error value
def geterror(errortype, pct, predicted, ms_test):
    if errortype == 1:
        av_error, av_pcterror = meanabserror(predicted, ms_test)
        
    elif errortype == 2:
        av_error, av_pcterror = meanerror(predicted, ms_test)
        
    elif errortype == 3:
        av_error, av_pcterror = rms(predicted, ms_test)
    
    if pct == True:
        return av_pcterror
    else:
        return av_error
    
#gets the mean absolute error (value and percentage)
def meanabserror(predicted, ms_data):
    error = []
    av_ms = mean(ms_data)
    points = len(predicted)
    
    for i in range(0,points):
        error.append(mean(abs(predicted[i] - ms_data[i])))
    
    av_error = mean(error)
    pcterror = error/av_ms*100
    av_pcterror = mean(pcterror)
    
#    print('Mean Absolute Error: '+str(av_error))
#    print('Percent Mean Absolute Error: '+str(av_pcterror))
    
    return av_error, av_pcterror

#gets the mean error (value and percentage)
def meanerror(predicted, ms_data):
    error = []
    av_ms = mean(ms_data)
    points = len(predicted)
    
    for i in range(0,points):
        error.append(mean(predicted[i] - ms_data[i]))
    
    av_error = mean(error)
    pcterror = error/av_ms*100
    av_pcterror = mean(pcterror)
    
#    print('Mean Error: '+str(av_error))
#    print('Percent Mean Error: '+str(av_pcterror))
    
    return av_error, av_pcterror

#gets the RMS error (value and percentage)
def rms(predicted, ms_data):
    error = []
    av_ms = mean(ms_data)
    points = len(predicted)
    
    for i in range(0, points):
        #error.append(sqrt(mean(square(predicted[i] - ms_data[i]))))
        error.append(square(predicted[i]-ms_data[i]))
    
    av_error = sqrt(mean(error))
    av_pcterror = av_error/av_ms*100
    
#    print('Root Mean Square Error: '+str(av_error))
#    print('Percent Root Mean Square Error: '+str(av_pcterror))
    
    return av_error, av_pcterror

#makes and saves (if desired) a parity plot of one slice
def parity(predicted, actual, slicetoshow, save):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(predicted, actual)
    plt.plot([0,.2], [0,.2], color = 'r', linestyle = '-')
    plt.title('Predicted vs Actual, Slice ' + str(slicetoshow))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save:
        saveplot('parity_slice_' + str(slicetoshow))
    else:
        plt.show() 
        
    return

#creates a single error map on a set scale
def slicedifcompare(difimage, actualimage, slicetoshow, save):
    
    #plots differences
    maxcolor, mincolor = -.05, .15
    tickrange = [-.05, 0, .05, .1, .15]
    fig = plt.figure(figsize=(8,4))
    
    ax1 = fig.add_subplot(121)
    fig.colorbar(ax1.pcolormesh(difimage, cmap = cm.viridis, vmin=mincolor, 
                                vmax=maxcolor), ticks=tickrange)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim(0, difimage.shape[1])
    ax1.set_ylim(0, difimage.shape[0])
    plt.title('Differences between Predicted and Actual', size = 11)
    
    #plots actual
    ax2 = fig.add_subplot(122)
    fig.colorbar(ax2.pcolormesh(actualimage, cmap = cm.viridis, vmin=mincolor, 
                                vmax=maxcolor), ticks=tickrange)
    ax2.set_aspect('equal', 'box')
    ax2.set_xlim(0, difimage.shape[1])
    ax2.set_ylim(0, difimage.shape[0])
    plt.title('Actual', size = 11)
    plt.suptitle('Slice ' + str(slicetoshow))
    
    if save:
        saveplot('diffcompare_slice_' + str(slicetoshow))
    else:
        plt.show() 
        
    return

def difzoomcompare(difimage, actualimage, slicetoshow, rep, fold, number, model, frac, 
                   dn, deg, inputs, showzoom=False, shownozoom=False, save,
                   maxcolor1=None, mincolor1=None):
    plt.close('all')
    frac = round(frac*100)
    
    if maxcolor1 == None:
        maxcolor1 = np.amax(actualimage)
    if mincolor1 == None:
        mincolor1 = 0
        
    if showzoom:
        maxcolor2 = np.amax(difimage)
        mincolor2 = np.amin(difimage)
    
    if showzoom and shownozoom:
        totalplts = 3
    elif showzoom or shownozoom:
        totalplts = 2
    else:
        totalplts = 1
        
    #plots actual
    n = 1
    if dn == 1:
        fig = plt.figure(figsize=(10,3))
    if dn == 3 or dn == 4:
        fig = plt.figure(figsize=(11.5,3))
    ax1 = fig.add_subplot(1,totalplts,n)
    tickrange = np.linspace(mincolor1, maxcolor1, num=5)
    fig.colorbar(ax1.pcolormesh(actualimage, cmap = cm.viridis, vmin=mincolor1, 
                                vmax=maxcolor1), ticks=tickrange)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim(0, difimage.shape[1])
    ax1.set_ylim(0, difimage.shape[0])
    ax1.set_xlabel('Actual', size = 11)
#    plt.title('Actual', size = 11)
    plt.suptitle('Data='+str(dn)+', degree='+str(deg)+', inputs='+str(inputs)
                +', model='+str(model)+', '+str(frac)+'% out, slice='
                +str(slicetoshow))
        
    #plots differences (no zoom)
    if shownozoom:
        n = n + 1
        tickrange = np.linspace(mincolor1, maxcolor1, num=5)
        ax2 = fig.add_subplot(1,totalplts,n)
        fig.colorbar(ax2.pcolormesh(difimage, cmap = cm.viridis, vmin=mincolor1, 
                                    vmax=maxcolor1), ticks=tickrange)
        ax2.set_aspect('equal', 'box')
        ax2.set_xlim(0, difimage.shape[1])
        ax2.set_ylim(0, difimage.shape[0])
        ax2.set_xlabel('Differences (orig scale)', size = 11)
#        plt.title('Differences (orig scale)', size = 11)         
        
    #plots differences (with zoom)
    if showzoom:
        n = n + 1
        tickrange = np.linspace(mincolor2, maxcolor2, num=5)
        ax2 = fig.add_subplot(1,totalplts,n)
        fig.colorbar(ax2.pcolormesh(difimage, cmap = cm.viridis, vmin=mincolor2, 
                                    vmax=maxcolor2), ticks=tickrange)
        ax2.set_aspect('equal', 'box')
        ax2.set_xlim(0, difimage.shape[1])
        ax2.set_ylim(0, difimage.shape[0])
        ax2.set_xlabel('Differences (zoom scale)', size = 11)
#        plt.title('Differences (zoom scale)', size = 11)
        

    if save:
        saveplot('dif_dn'+str(dn)+'n'+str(number)+'_r'+str(rep)+'_f'+str(fold)+'_s'+str(slicetoshow))
    else:
        plt.show() 

#creates a single error map on a custom scale
def difzoom(difimage, save=False, maxcolor=None, mincolor=None, slicetoshow=None):
    #plots differences between predicted and actual on a different scale
    
    if maxcolor == None:
        maxcolor = np.amax(difimage)
    if mincolor == None:
        mincolor = np.amin(difimage)
    
    if slicetoshow != None:
        name = 'Differences between Predicted and Actual, Slice ' + str(slicetoshow)
    else:
        name = 'Differences between Predicted and Actual'
        
    tickrange = np.linspace(mincolor, maxcolor, num=5)
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111)
    fig.colorbar(ax.pcolormesh(difimage, cmap = cm.viridis, vmin=mincolor, 
                                vmax=maxcolor), ticks=tickrange)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(0, difimage.shape[1])
    ax.set_ylim(0, difimage.shape[0])
    plt.title(name, size = 11)
    
    if slicetoshow != None:
        if save:
            saveplot('diffzoom_slice_' + str(slicetoshow))
        else:
            plt.show() 
    elif slicetoshow == None:
        if save:
            saveplot('diffzoom')
        else:
            plt.show() 
        
    return

#creates a parity plot using all test data
def allparity(predicted, ms_test, rep, fold, number, model, frac, dn, deg, inputs, save):
    plt.close('all')
    frac = round(frac*100)
    overallmax = max(np.amax(predicted), np.amax(ms_test))
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(1,1,1)
    plt.scatter(predicted, ms_test)
    plt.plot([0,overallmax], [0,overallmax], color = 'r', linestyle = '-')
    plt.title('Data='+str(dn)+', degree='+str(deg)+', inputs='+str(inputs)
                +', model='+str(model)+', '+str(frac)+'% out parity plot')
    plt.xlabel('Predicted')
    plt.ylabel('Actual') 
    plt.tight_layout() 
        
    if save:
        saveplot('par_dn'+str(dn)+'_n'+str(number)+'_r'+str(rep)+'_f'+str(fold))
    else:
        plt.show() 
        
    return

#creates one error map for each slice removed--for slicewise cross val
def allslicedifcompare(predicted, ms_test, slicesout, shape, trimrows, save=False):
    #creates difference image map
    numslices = len(slicesout)
    if numslices == 0:
        numslices = 20
        slicesout = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    
    #plots differences
    if predicted.shape != shape:
        predictedarray = vectortoimage(predicted, [shape[0]-2*trimrows, 
                                                shape[1]-2*trimrows, numslices])
        msarray = vectortoimage(ms_test, [shape[0]-2*trimrows, 
                                                shape[1]-2*trimrows, numslices])
    else:
        predictedarray = predicted
        msarray = ms_test
        
    maxcolor, mincolor = -.05, .15
    tickrange = [-.05, 0, .05, .1, .15]
    fig = plt.figure(figsize=(8,4*numslices))
    for i in range(0,numslices):
        actualimage = msarray[:,:,i]
        difimage = actualimage - predictedarray[:,:,i]
        ax1 = fig.add_subplot(numslices,2,(i+1)*2-1)
        fig.colorbar(ax1.pcolormesh(difimage, cmap = cm.viridis, vmin=mincolor, 
                                    vmax=maxcolor), ticks=tickrange)
        ax1.set_aspect('equal', 'box')
        ax1.set_xlim(0, difimage.shape[1])
        ax1.set_ylim(0, difimage.shape[0])
        plt.title('Differences, slice '+str(slicesout[i]), size = 11)
        
        #plots actual
        ax2 = fig.add_subplot(numslices,2,(i+1)*2)
        fig.colorbar(ax2.pcolormesh(actualimage, cmap = cm.viridis, vmin=mincolor, 
                                    vmax=maxcolor), ticks=tickrange)
        ax2.set_aspect('equal', 'box')
        ax2.set_xlim(0, difimage.shape[1])
        ax2.set_ylim(0, difimage.shape[0])
        plt.title('Actual, slice '+str(slicesout[i]), size = 11)

    plt.tight_layout()
    
    if save:
        saveplot('diffcompare')
    else:
        plt.show()    
    
    return 

#takes a vector and transforms it back into array (image) form
def vectortoimage(data, shape):
    image = np.zeros(shape)
    count = 0
    for k in range(0, shape[2]):
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                image[i,j,k] = data[count]
                count = count + 1
    
    return image


#creates error maps for each slice on a custom scale--for single-slice k fold cross val
def alldifzoom(predictedarray, ms_data_trimmed, shape, maxcolor=None, mincolor=None, save=False):
    #plots differences between predicted and actual on a different scale
    dif = predictedarray - ms_data_trimmed
    if maxcolor == None:
        maxcolor = np.amax(dif)
    if mincolor == None:
        mincolor = np.amin(dif)
        
    tickrange = np.linspace(mincolor, maxcolor, num=5)
    fig = plt.figure(figsize=(8,37))
    for i in range(0,20):
        difimage = dif[:,:,i]
        ax = fig.add_subplot(10,2,i+1)
#        difimage, actualimage = getimages(predictedarray[i], 
#                                          ms_data_trimmed[:][:][i], i, shape)
        fig.colorbar(ax.pcolormesh(difimage, cmap = cm.viridis, vmin=mincolor, 
                                    vmax=maxcolor), ticks=tickrange)
        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, difimage.shape[1])
        ax.set_ylim(0, difimage.shape[0])
        plt.title('Slice '+str(i))

    plt.tight_layout()
    
    if save:
        saveplot('diffzoom')
    else:
        plt.show()
    
    return
 
#rearranges vectors into arrays and gets error maps       
def getimages(predicted, ms_data_trimmed, slicestoshow, shapes, inputs):
    trimrows = math.floor(np.sqrt(inputs)/2)
    difimagelist = []
    actualimagelist = []
    
    #rearrange vector into array
    for n in range(len(slicestoshow)):
        shape = shapes[n]
        elements = (shape[0]-2*trimrows)*(shape[1]-2*trimrows)
        predictedimage = np.reshape(predicted[0:elements], (shape[0]-2*trimrows, shape[1]-2*trimrows))
        actualimage = np.reshape(ms_data_trimmed[0:elements], (shape[0]-2*trimrows, shape[1]-2*trimrows))
        predicted = np.delete(predicted, range(elements))
        ms_data_trimmed = np.delete(ms_data_trimmed, range(elements))
        
        #obtain difference array
        difimage = predictedimage - actualimage
        
        difimagelist.append(difimage)
        actualimagelist.append(actualimage)
        
    return difimagelist, actualimagelist

#saves the plot to a timestamped folder
def saveplot(name):
    date = datetime.now()
#    folderdate = date.strftime('%Y-%m-%d figures')
#    mkdir_p(folderdate)
    filedate = date.strftime(name + '_%Y-%m-%d_%H;%M;%S')
    plt.savefig(filedate + '.png')
    return

#makes a timestamped folder
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
    return

#creates histograms of error values
def errorhist(error, bins, binmin, binmax, save, modeltype, valtype, errortype,
              fracin, deg, inputs):
    plt.clf()
    plt.hist(error, bins = bins, range = (binmin, binmax))
    if valtype == 1:
        val = 'Crossval type: slice k fold, in order'
    elif valtype == 2:
        val = 'Crossval type: random block'
    elif valtype == 3:
        val = 'Crossval type: random point'
    elif valtype == 4:
        val = 'Crossval type: gridded random block'
    elif valtype == 5:
        val = 'Crossval type: slice k fold, random order' 
        
    if modeltype == 1:
        mod = 'Polynomial, degree: ' + str(deg) + ' inputs: ' + str(inputs) + ' wo cross'
    elif modeltype == 2:
        mod = 'Polynomial, degree: ' + str(deg) + ' inputs: ' + str(inputs) + ' w cross'
    elif modeltype == 3:
        mod = 'Ridge regression, inputs: ' + str(inputs)
    elif modeltype == 4:
        mod = 'Kernel ridge regression, inputs: ' + str(inputs)
        
    if errortype == 1:
        et = 'Mean absolute error, '
    elif errortype == 2:
        et = 'Mean error,  '
    elif errortype == 3:
        et = 'RMS error, '
        
    plt.title(et + mod + '\n' + val + ', fraction in: ' + str(fracin))
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    
    if save == True:
        saveplot('errorhist'+ str(errortype))
    else:
        plt.show()
    return

def nCk(n,k):
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1))