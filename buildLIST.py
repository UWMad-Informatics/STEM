#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:39:26 2017

@author: aidan
"""
from numpy import sqrt
import math
import numpy as np
import random
import traintestLIST
from scipy.ndimage.filters import gaussian_filter
from sklearn.preprocessing import PolynomialFeatures

#Convolves an image by a Gaussian function
def blur(convdata, sigma=4, truncate=2, norm=True):
    numofslices = len(convdata)
    blurreddata = []
    total = 0
    for s in range(numofslices):
        image = convdata[s]
        total = total + np.sum(image)
        blur = gaussian_filter(image, sigma, truncate=truncate)
        blurreddata.append(blur)
        
    #forces blur to conserve intensity--normalizes it to the sum of the 
    #original array
    if norm == True:
        blurreddata = normalize(blurreddata, total)
        
    return blurreddata

#normalizes an array to a certain value
def normalize(array, val):
    total = 0
    numofslices = len(array)
    for s in range(numofslices):
        total = total + np.sum(array[s])
        
    normed = []
    for s in range(numofslices):
        normedslice = array[s] * (val/total)
        normed.append(normedslice)
        
    return normed

#def scalefit(conv, ms, inputs, deg, cross=False, logterm=False):
#    conv = np.reshape(conv, (-1))
#    ms = np.reshape(ms, (-1))
#    model = traintestLIST.poly(conv, ms, deg, inputs, cross, logterm)
#    if cross == False and logterm == False:
#        modeltype = 1
#    elif cross == False and logterm == True:
#        modeltype = 6
#    elif cross == True and logterm == False:
#        modeltype = 2
#    else:
#        modeltype = 7
#    
#    predicted, time = traintestLIST.predict(conv, deg, inputs, model, modeltype)
#    return predicted

#Creates a trains a model according to specs: model degree, number of inputs, 
#type of model, type of cross validation, fraction of data left in, size of 
#block. Repeats a set number of times. Returns a list of models, a list of test
#lists, and a list of train lists.
def modelkfoldrep(multislice_data, convolution_data, deg, inputs, modeltype, 
               crossvaltype, fracin, blocksize, repeat):
    #modeltype 1: polynomial no crossterms
    #modeltype 2: polynomial crossterms
    #modeltype 3: ridge regression
    #modeltype 4: kernel ridge regression
    #modeltype 5: gaussian process regression
    #
    #crossvaltype 1: slicewise, in order k fold
    #crossvaltype 2: random blocks
    #crossvaltype 3: random pixels, no buffer
    #crossvaltype 4: grid blocks, k fold
    #crossvaltype 5: slicewise, random k fold

    if inputs == 1:
        trimrows = 0
    if inputs == 9:
        trimrows = 1
    if inputs == 25:
        trimrows = 2
    if inputs == 49:
        trimrows = 3
        
    xtestlistoflists = []
    mstestlistoflists = []
    modellistoflists = []
    slicesinfoldslistoflists = []
    traincenterslist = []
    testcenterslist = []
    outnumlist = []
    
    for j in range(0, repeat):  
        slicesinfolds = []
        if crossvaltype == 1:
            inorder = True
        else:
            inorder = False
        
        if crossvaltype == 1 or crossvaltype == 5:
            xtrainlist, mstrainlist, xtestlist, mstestlist, traincenters, testcenters, outnum, slicesinfolds = slicekfold(convolution_data, 
                                             multislice_data, fracin, inputs, trimrows, inorder)
        if crossvaltype == 4:
            xtrainlist, mstrainlist, xtestlist, mstestlist, traincenters, testcenters, outnum = gridblockout(convolution_data, 
                                            multislice_data, fracin, inputs, blocksize)
            
        if crossvaltype == 3:
            xtrainlist, mstrainlist, xtestlist, mstestlist, traincenters, testcenters, outnum = randompixelkfold(convolution_data,
                                                                                                                multislice_data, 
                                                                                                                 fracin, inputs)
                        
        i = 0
        modellist = []
        for X_train in xtrainlist:
            ms_train = mstrainlist[i]
            if modeltype == 1:
                #Builds a polynomial model of specified degree without 
                #crossterms and log term
                model = traintestLIST.poly(X_train, ms_train, deg, inputs, False, False)   
            if modeltype == 2:
                #Builds a polynomial model of specified degree with crossterms 
                #and without log term
                model = traintestLIST.poly(X_train, ms_train, deg, inputs, True, False)
            if modeltype == 3:
                #Builds a RR model
                model = traintestLIST.rr(X_train, ms_train)      
            if modeltype == 4:
                #Builds a KRR model
                model = traintestLIST.krr(X_train, ms_train)
            if modeltype == 5:
                #Builds a gaussian process regression model
                model = traintestLIST.gpr(X_train, ms_train)
            if modeltype == 6: 
                #Builds a polynomial model of specified degree without 
                #crossterms and with log term
                model = traintestLIST.poly(X_train, ms_train, deg, inputs, False, True) 
            if modeltype == 7:
                #Builds a polynomial model of specified degree without 
                #crossterms and with log term
                model = traintestLIST.poly(X_train, ms_train, deg, inputs, True, True) 
            modellist.append(model)
            i = i + 1
            
        xtestlistoflists.append(xtestlist)
        mstestlistoflists.append(mstestlist)
        modellistoflists.append(modellist)
        slicesinfoldslistoflists.append(slicesinfolds)
        traincenterslist.append(traincenters)
        testcenterslist.append(testcenters)
        outnumlist.append(outnum)
    
    traincenters = np.mean(traincenterslist)
    testcenters = np.mean(testcenterslist)
    outnum = np.mean(outnumlist)
    
    traincentersperc = traincenters/(traincenters+testcenters+outnum)
    testcentersperc = testcenters/(traincenters+testcenters+outnum)
    outnumperc = outnum/(traincenters+testcenters+outnum)
    
    return modellistoflists, xtestlistoflists, mstestlistoflists, traincentersperc, testcentersperc, outnumperc, slicesinfoldslistoflists

#Creates a trains a model according to specs: model degree, number of inputs, 
#type of model, type of cross validation, fraction of data left in, size of 
#block. Does not repeat. Returns a model, a test set, and the number of 
#test and train centers. 
def modelkfold(multislice_data, convolution_data, deg, inputs, modeltype, 
               crossvaltype, fracin, blocksize):
    #modeltype 1: polynomial no crossterms
    #modeltype 2: polynomial crossterms
    #modeltype 3: ridge regression
    #modeltype 4: kernel ridge regression
    #modeltype 5: gaussian process regression
    #
    #crossvaltype 1: slicewise, in order k fold
    #crossvaltype 2: random blocks NOT UP TO DATE
    #crossvaltype 3: random pixels, no buffer
    #crossvaltype 4: grid blocks, k fold
    #crossvaltype 5: slicewise, random k fold

    if inputs == 1:
        trimrows = 0
    if inputs == 9:
        trimrows = 1
    if inputs == 25:
        trimrows = 2
    if inputs == 49:
        trimrows = 3
        
    traincenterslist = []
    testcenterslist = []
    outnumlist = []
    slinflist = []
    
    if crossvaltype == 1:
        inorder = True
    else:
        inorder = False
    
    if crossvaltype == 1 or crossvaltype == 5:
        xtrainlist, mstrainlist, xtestlist, mstestlist, traincenters, testcenters, outnum, slinflist = slicekfold(convolution_data, 
                                         multislice_data, fracin, inputs, trimrows, inorder)
    if crossvaltype == 4:
        xtrainlist, mstrainlist, xtestlist, mstestlist, traincenters, testcenters, outnum = gridblockout(convolution_data, 
                                        multislice_data, fracin, inputs, blocksize)
        
    if crossvaltype == 3:
        xtrainlist, mstrainlist, xtestlist, mstestlist, traincenters, testcenters, outnum = randompixelkfold(convolution_data,
                                                                                                            multislice_data, 
                                                                                                             fracin, inputs)
                    
    i = 0
    modellist = []
    for X_train in xtrainlist:
        ms_train = mstrainlist[i]
        if modeltype == 1:
            #Builds a polynomial model of specified degree without 
            #crossterms and log term
            model = traintestLIST.poly(X_train, ms_train, deg, inputs, False, False)   
        if modeltype == 2:
            #Builds a polynomial model of specified degree with crossterms 
            #and without log term
            model = traintestLIST.poly(X_train, ms_train, deg, inputs, True, False)
        if modeltype == 3:
            #Builds a RR model
            model = traintestLIST.rr(X_train, ms_train)      
        if modeltype == 4:
            #Builds a KRR model
            model = traintestLIST.krr(X_train, ms_train)
        if modeltype == 5:
            #Builds a gaussian process regression model
            model = traintestLIST.gpr(X_train, ms_train)
        if modeltype == 6: 
            #Builds a polynomial model of specified degree without 
            #crossterms and with log term
            model = traintestLIST.poly(X_train, ms_train, deg, inputs, False, True) 
        if modeltype == 7:
            #Builds a polynomial model of specified degree without 
            #crossterms and with log term
            model = traintestLIST.poly(X_train, ms_train, deg, inputs, True, True) 
        modellist.append(model)
        i = i + 1
        
    traincenterslist.append(traincenters)
    testcenterslist.append(testcenters)
    outnumlist.append(outnum)
    
    traincenters = np.mean(traincenterslist)
    testcenters = np.mean(testcenterslist)
    outnum = np.mean(outnumlist)
    
    traincentersperc = traincenters/(traincenters+testcenters+outnum)
    testcentersperc = testcenters/(traincenters+testcenters+outnum)
    outnumperc = outnum/(traincenters+testcenters+outnum)
    
    return modellist, xtestlist, mstestlist, slinflist, traincentersperc, testcentersperc, outnumperc

#Removes a certain number of slices from the dataset, randomly or in order. 
#Repeats for k-fold.
def slicekfold(convdata, msdata, fracin, inputs, trimrows, inorder):
    numofslices = len(convdata)
    slicenum = round((1-fracin)*numofslices)
    slices = []
    slices2 = []
    xtrainlist = []
    xtestlist = []
    mstrainlist = []
    mstestlist = []
    testcenterslist = []
    traincenterslist = []
    outnumlist = []
    slicesinfold = []
    
    for i in range(0, numofslices):
        slices.append(i)
        slices2.append(i)
    
    if slicenum == 0:
        repeatnum = 1
    else:
        repeatnum = math.ceil(numofslices/slicenum)
    
    for i in range(0, repeatnum):
        slicesinfold.append([])
        slicesout = []
        
        #chooses slices to remove randomly from a list of slices that haven't been 
        #used yet. If all have been used, repeats are allowed, but the same slice
        #can't be used twice in the same fold
        if inorder == False:
            for j in range(0, slicenum):
                if not slices:
                    slices = slices2
                    
                index = random.choice(slices)
                while index in slicesinfold[i]:
                    index = random.choice(slices)
                    
                slicesinfold[i].append(index)
                slices.remove(index)
                slicesout.append(index)
        
        #uses list of slices in order instead
        elif inorder == True:
            for j in range(0, slicenum):
                if not slices:
                    slices = slices2
                index = slices[0]
                slices.remove(index)
                slicesout.append(index)
                slicesinfold[i].append(index)
        
        elements = 0
        totalelements = 0
        for s in range(numofslices):
            shape = convdata[s].shape
            elements = elements + (shape[0]-trimrows*2) * (shape[1]-trimrows*2)
            totalelements = totalelements + shape[0]*shape[1]
#        elements = (shape[0]-trimrows*2) * (shape[1]-trimrows*2)  
            
        numelout = 0
        for s in slicesout:
            shape = convdata[s].shape
            numelout = numelout + (shape[0]-trimrows*2) * (shape[1]-trimrows*2)
        #Builds vector of all slices excluding specified slices   
        numout = len(slicesout)
        X_train = np.zeros([elements-numelout, inputs])
        X_test = np.zeros([numelout, inputs])
        ms_train = np.zeros([elements-numelout,1])
        ms_test = np.zeros([numelout,1])
        traincount = 0;
        slicecount = 0;
    
        for k in range (0, numofslices):
            for i in range (trimrows, convdata[k].shape[0]-trimrows):
                for j in range (trimrows, convdata[k].shape[1]-trimrows):
                    if (k in slicesout) == False:                   
                        #convolution data to vector excluding 1 slice
                        X_train[traincount, :] = getregiondata(inputs, convdata, i, j, k)
                        ms_train[traincount, 0] = msdata[k][i,j]
                        traincount = traincount + 1
                        
                    if k in slicesout:
                        X_test[slicecount, :] = getregiondata(inputs, convdata, i, j, k)
                        ms_test[slicecount, 0] = msdata[k][i,j]
                        slicecount = slicecount + 1
        if numout == 0:
            X_test = X_train
            ms_test = ms_train
            
        traincenters = len(ms_train)
        testcenters = len(ms_test)
        outnum = totalelements - traincenters - testcenters
        
        xtestlist.append(X_test)
        xtrainlist.append(X_train)
        mstestlist.append(ms_test)
        mstrainlist.append(ms_train)
        
        traincenterslist.append(traincenters)
        testcenterslist.append(testcenters)
        outnumlist.append(outnum)
    
    traincenters = np.mean(traincenterslist)
    testcenters = np.mean(testcenterslist)
    outnum = np.mean(outnumlist)
             
    return xtrainlist, mstrainlist, xtestlist, mstestlist, traincenters, testcenters, outnum, slicesinfold

#Removes a given portion of random pixels from the data set--does not leave
#any buffer zone around them. Repeats to give k-fold.
def randompixelkfold(convdata, msdata, fracin, inputs):
    numofslices = len(convdata)
    totalelements = 0
    noedgeels = 0
    buffer = int(math.sqrt(inputs)-1)
    trimrows = int(buffer/2)
    for s in range(numofslices):
        shape = convdata[s].shape
        totalelements = totalelements + shape[0]*shape[1]
        noedgeels = noedgeels + (shape[0]-buffer)*(shape[1]-buffer)
    numleavein = int(fracin*noedgeels)
    numtakeout = noedgeels-numleavein
    numfolds = int(math.ceil(noedgeels/numtakeout))
    
#    gridrows = shape[0]-buffer
#    gridcols = shape[1]-buffer
    xtestlist = []
    xtrainlist= []
    mstestlist = []
    mstrainlist = []
    testcenterslist = []
    traincenterslist = []
    outnumlist = []
    
    rands = []
    rands2 = []
    randsinfold = []
    folds = 0
    
    for i in range(noedgeels):
        rands.append(i)
        rands2.append(i)

    #places pixels
    for n in range(0, numfolds):
        randsinfold.append([])
        out = []
        for s in range(len(convdata)):
            shape = convdata[s].shape
            outslice = np.zeros([shape[0], shape[1]])
            out.append(outslice)
        folds = folds + 1
        
        for m in range(0, numtakeout):
            if not rands:
                rands = rands2
            #generates list of random indices that haven't already been used
            randindex = random.choice(rands)
            while randindex in randsinfold[folds-1]:
                randindex = random.choice(rands)
            randsinfold[folds-1].append(randindex)
            rands.remove(randindex)
            randloc = pixelindextoloc(randindex, trimrows, convdata)
            #pulls out data for the current random pixel in the list. Marks these
            #locations in the out array with 2. Also marks invalid zones in out array 
            #with 1. Valid squares are marked with 0.
            out[randloc[2]][randloc[0], randloc[1]] = 2

        invalidarray, invalidcount = findinvalidnobuffer(out, trimrows)
        validcount = totalelements - invalidcount                
        #now assign training and test data   
        X_test = np.zeros([numtakeout, inputs])
        ms_test = np.zeros([numtakeout, 1])                  
        X_train = np.zeros([validcount, inputs])
        ms_train = np.zeros([validcount, 1])        
    
        testcount = 0
        traincount = 0
        for k in range(0, numofslices):
            currslice = invalidarray[k]
            for i in range(0, convdata[k].shape[0]):
                for j in range(0, convdata[k].shape[1]):
                    if currslice[i,j] == 2:
                        X_test[testcount, :] = getregiondata(inputs, convdata, i,j,k)
                        ms_test[testcount, :] = msdata[k][i,j]
                        testcount = testcount + 1
                    elif currslice[i,j] == 0:
                        X_train[traincount, :] = getregiondata(inputs, convdata, i,j,k)
                        ms_train[traincount, :] = msdata[k][i,j]
                        traincount = traincount + 1
            
        traincenters = len(ms_train)
        testcenters = len(ms_test)
        outnum = totalelements - noedgeels
                        
        xtestlist.append(X_test)
        xtrainlist.append(X_train)
        mstestlist.append(ms_test)
        mstrainlist.append(ms_train)
        
        traincenterslist.append(traincenters)
        testcenterslist.append(testcenters)
        outnumlist.append(outnum)

    traincenters = np.mean(traincenterslist)
    testcenters = np.mean(testcenterslist)
    outnum = np.mean(outnumlist)
    
    return xtrainlist, mstrainlist, xtestlist, mstestlist, traincenters, testcenters, outnum

#removes square blocks of points of a specified size. Places the blocks randomly
#in a set grid calculated by using the block size. Repeats for k-fold.
def gridblockout(convdata, msdata, fracin, inputs, blocksize):
    numofslices = len(convdata)
    buffer = int(math.sqrt(inputs)-1)
    gridrows = np.zeros([numofslices])
    gridcols = np.zeros([numofslices])
    trimrowsup = np.zeros([numofslices])
    trimcolsleft = np.zeros([numofslices])
    totalelements = 0
    noedgeels = 0
    for s in range(numofslices):
        shape = convdata[s].shape
        totalelements = totalelements + shape[0]*shape[1]
        noedgeels = noedgeels + (shape[0]-buffer)*(shape[1]-buffer)
        gridrows[s] = math.floor((shape[0]-buffer)/blocksize)
        trimrowsup[s] = math.floor(((shape[0]-buffer)%blocksize)/2)
        gridcols[s] = math.floor((shape[1]-buffer)/blocksize)
        trimcolsleft[s] = math.floor(((shape[1]-buffer)%blocksize)/2)

    numleavein = int(fracin*noedgeels)
    
    totalblocks = 0
    outblank = []
    for s in range(numofslices):
        totalblocks = totalblocks + int(gridrows[s]*gridcols[s])
        outblank.append(np.zeros([convdata[s].shape[0], convdata[s].shape[1]]))
        
    xtestlist = []
    xtrainlist= []
    mstestlist = []
    mstrainlist = []
    testcenterslist = []
    traincenterslist = []
    outnumlist = []
    
    #blocksout = np.zeros([gridrows, gridcols, shape[2]])
    kfolddone = False
    #creates a list of indices of the blocks
    rands = []
    rands2= []
    for i in range(0, totalblocks):
        rands.append(i)
        rands2.append(i)
    
    #if either gridrows or gridcols is 0, then the block size is too big.
    if 0 in gridrows or 0 in gridcols:
        kfolddone = True
    
    folds = 0
    randsinfold = []
    
    while kfolddone == False:
        folds = folds + 1
        numblocks = 0
        randsinfold.append([])
        out = []
        for s in range(numofslices):
            out.append(np.zeros([convdata[s].shape[0], convdata[s].shape[1]]))
       
        done = False
        #places blocks 
        while done == False:
            #increments number of blocks by one
            numblocks = numblocks + 1
            #generates a random index that hasn't already been used
            if len(rands) == 0:
                rands = rands2
                kfolddone = True
                
            randindex = random.choice(rands)
            while randindex in randsinfold[folds-1]:
                randindex = random.choice(rands)
            randsinfold[folds-1].append(randindex)
            rands.remove(randindex)
            randblockloc = blockindextoloc(randindex, gridrows, gridcols)
            randloc = blockloctogridloc(randblockloc[0], randblockloc[1], 
                                        randblockloc[2], blocksize, buffer, 
                                                    trimrowsup, trimcolsleft)
            
            #pulls out data for the current random block in the list. Marks these
            #locations in the out array with 2. Also marks invalid zones in out array 
            #with 1. Valid squares are marked with 0.
            for ii in range(0, blocksize):
                for jj in range(0, blocksize):
                    k = randloc[2]
                    outslice = out[k]
                    outslice[randloc[0]+ii, randloc[1]+jj] = 2
                    out[k] = outslice
            #determines elements that are invalid and whether to continue loop.
            #stops loop when enough elements have been taken out or when there
            #are no more valid blocks remaining.
            invalidarray, invalidcount = findinvalid(out, buffer)
            validcount = totalelements-invalidcount
            if numleavein >= validcount:
                done = True
                        
        #now assign training and test data 
        testnum = numblocks * blocksize * blocksize
        X_test = np.zeros([testnum, inputs])
        ms_test = np.zeros([testnum, 1])                  
        X_train = np.zeros([validcount, inputs])
        ms_train = np.zeros([validcount, 1])        
    
        #assigns training data by checking out matrix for squares marked valid
        traincount = 0
        for k in range(0, numofslices):
            currslice = invalidarray[k]
            shape = currslice.shape
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    if currslice[i,j] == 0:
                        X_train[traincount, :] = getregiondata(inputs, convdata, i,j,k)
                        ms_train[traincount, :] = msdata[k][i,j]
                        traincount = traincount + 1
                        
        testcount = 0
        #for each block removed, add its data to the test matrix. Allows for 
        #repeated blocks, which just searching the out matrix for squares marked
        #two would not do.
        for n in randsinfold[folds-1]:
            randblockloc = blockindextoloc(n, gridrows, gridcols)
            randloc = blockloctogridloc(randblockloc[0], randblockloc[1], randblockloc[2], 
                                        blocksize, buffer, trimrowsup, trimcolsleft)
            for ii in range(0, blocksize):
                for jj in range(0, blocksize):
                    i = randloc[0] + ii
                    j = randloc[1] + jj
                    k = randloc[2]
                    X_test[testcount, :] = getregiondata(inputs, convdata, i, j, k)
                    ms_test[testcount, :] = msdata[k][i,j]
                    testcount = testcount + 1
                        
        #determines whether all data has been part of test set and therefore whether 
        #to continue loop
        if not rands:
            kfolddone = True
            
        traincenters = len(ms_train)
        testcenters = len(ms_test)
        outnum = totalelements - traincenters - testcenters
                        
        xtestlist.append(X_test)
        xtrainlist.append(X_train)
        mstestlist.append(ms_test)
        mstrainlist.append(ms_train)
        
        traincenterslist.append(traincenters)
        testcenterslist.append(testcenters)
        outnumlist.append(outnum)
    
    if 0 in gridcols or 0 in gridrows:
        traincenters = 0
        testcenters = 0
        outnum = 0
    else:
        traincenters = np.mean(traincenterslist)
        testcenters = np.mean(testcenterslist)
        outnum = np.mean(outnumlist)
    
    return xtrainlist, mstrainlist, xtestlist, mstestlist, traincenters, testcenters, outnum

#Given a dataset and the index of a point, gets the point and a number of its
#nearest neighbors
def getregiondata(inputs, data, i, j, k):
    index = math.floor(sqrt(inputs)/2)
    datalist = np.zeros(inputs)
    count = 0
    for ii in range (i-index, i+index+1):
        for jj in range (j-index, j+index+1):
            dataslice = data[k]
            datalist[count] = dataslice[ii,jj]
            count = count + 1
    return datalist

#finds points in the array which cannot be centers. Does not leave any buffer
#zone around points (so is used for the random pixel cross val)
def findinvalidnobuffer(out, trimrows):
    numofslices = len(out)
    trimrows = int(trimrows)
    totalelements = 0
    for k in range(0, numofslices):
        currslice = out[k]
        shape = currslice.shape
        totalelements = totalelements + shape[0]*shape[1]
        for i in range(0, trimrows):
            for j in range(0, shape[1]):
                if currslice[i][j] == 0:
                    currslice[i][j] = 1
                if currslice[shape[0]-1-i][j] == 0:
                    currslice[shape[0]-1-i][j] = 1
        for j in range(0, trimrows):
            for i in range(0, shape[0]):
                if currslice[i][j] == 0:
                    currslice[i][j] = 1
                if currslice[i][shape[1]-j-1] == 0:
                    currslice[i][shape[1]-j-1] = 1
        out[k] = currslice
                   
    validcount = 0
    for k in range(0, numofslices):
        currslice = out[k]
        shape = currslice.shape
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                if currslice[i][j] == 0:
                    validcount = validcount + 1
                    
    invalidcount = totalelements - validcount
    return out, invalidcount

#Finds points in the array that cannot be centers, putting a buffer zone around
#taken points.
def findinvalid(out, buffer):
    numofslices = len(out)
    invalidarray = []
    for s in range(numofslices):
        currshape = out[s].shape
        invalidarray.append(np.zeros([currshape[0], currshape[1]]))

    invalidcount = 0
    edgedist = int(buffer/2)
    
    for k in range(0, numofslices):
        shape = out[k].shape
        outslice = out[k]
        invalidslice = invalidarray[k]
        #fills in squares that are invalid because of proximity to edge                                
        for i in range(0, shape[0]):
            for j in range(shape[1]-edgedist, shape[1]):
                if invalidslice[i,j] == 0:
                    invalidslice[i,j] = 1
                    invalidcount = invalidcount + 1
            for j in range(0, edgedist):
                if invalidslice[i,j] == 0:
                    invalidslice[i,j] = 1
                    invalidcount = invalidcount + 1
                            
        for j in range(0, shape[1]):
            for i in range(0, edgedist):
                if invalidslice[i,j] == 0:
                    invalidslice[i,j] = 1
                    invalidcount = invalidcount + 1
            for i in range(shape[0]-edgedist, shape[0]):
                if invalidslice[i,j] == 0:
                    invalidslice[i,j] = 1
                    invalidcount = invalidcount + 1
                
        #fills in squares that are invalid because of proximity to block
        for j in range(edgedist, shape[1]-edgedist):
            for i in range(edgedist, shape[0]-edgedist):
                if outslice[i,j] == 2:
                    #marks the centerpoint with a 2. These points are marked as invalid.
                    if invalidslice[i,j] == 0:
                        invalidcount = invalidcount + 1
                    invalidslice[i,j] = 2
                    
                    if buffer != 0:
                        starti = i-buffer
                        startj = j-buffer
                        endi = i+buffer+1
                        endj = j+buffer+1
                        if starti < 0:
                            starti = 0
                        if startj < 0:
                            startj = 0
                        if endi >= shape[0]:
                            endi = shape[0]-1
                        if endj >= shape[1]:
                            endj = shape[1]-1
                        
                        #goes and finds surrounding points, marks them with a 1
                        for ii in range(starti, endi):
                            for jj in range(startj, endj):
                                if invalidslice[ii,jj] == 0:
                                    invalidslice[ii,jj] = 1
                                    invalidcount = invalidcount + 1
        invalidarray[k] = invalidslice
                
    return invalidarray, invalidcount

def pixelindextoloc(index, trimrows, convdata):
    foundslice = False
    totalel = 0
    k = 0
    while foundslice == False:
        currslice = convdata[k]
        elinslice = (currslice.shape[0]-trimrows*2)*(currslice.shape[1]-trimrows*2)
        totalel = totalel + elinslice
        if totalel > index:
            foundslice = True
        else:
            k = k+1
    shape = convdata[k].shape
    index = int(index - (totalel-elinslice))
    i = int(math.floor(index/(shape[1]-2*trimrows))) + trimrows
    index = index % (shape[1] - 2*trimrows)
    j = int(index) + trimrows
    return[i,j,k]
    
#Converts a random block index to a location in the block grid.
def blockindextoloc(index, gridrows, gridcols):
    foundslice = False
    totalblocks = 0
    k = 0
    while foundslice == False:
        rows = gridrows[k]
        cols = gridcols[k]
        blocksinslice = rows * cols
        totalblocks = totalblocks + blocksinslice
        if totalblocks > index:
            foundslice = True
        else:
            k = k+1
    index = int(index - (totalblocks-blocksinslice))
    i = int(math.floor(index/cols))
    index = index % cols
    j = int(index)
    return [i,j,k]

#Takes a location in the block grid and finds its upper left corner in the
#actual grid.
def blockloctogridloc(i, j, k, blocksize, buffer, trimrowsup, trimcolsleft):
    trimup = trimrowsup[k]
    trimleft = trimcolsleft[k]
    i = int(i*blocksize + buffer/2 + trimup)
    j = int(j*blocksize + buffer/2 + trimleft)
    return [i, j, k]

#recursively builds a generator that yields all possible combinations of a 
#certain number (length) in the provided list (elements)
def combos(elements, length):
    for i in range(len(elements)):
        if length == 1:
            yield (elements[i],)
        else:
            for next in combos(elements[i+1:len(elements)], length-1):
                yield (elements[i],) + next

#uses the generator built by combos to get the list of combinations
def choose(l, k):
    return list(combos(l, k))

def findtermnums(deg, inputs, modeltype):
    X_train = np.ones([inputs, inputs])
    if modeltype == 1:
        crossterms = False
        logterm = False
        terms = inputs * deg + 1
    elif modeltype == 2:
        crossterms = True
        logterm = False
    elif modeltype == 6:
        crossterms = False
        logterm = True
        terms = (inputs + 1) * deg + 1
    elif modeltype == 7:
        crossterms = True
        logterm = True
    else:
        crossterms = False
        terms = 0

    if crossterms == True:
        if logterm == True:
            X_train = traintestLIST.addlog(X_train)
        poly = PolynomialFeatures(degree=deg)
        poly.fit_transform(X_train)
        terms = poly.n_output_features_
    return terms
