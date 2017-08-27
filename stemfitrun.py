#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:40:48 2017

@author: aidan
"""

import buildLIST
import analyzeLIST
import traintestLIST
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.io import loadmat
from PIL import Image

def stemrun(datanumber, fracin, deg, inputs, crossvaltype, blocksize, 
            modeltype, repeats, name, blurring, save, number):

    #modeltype 1: polynomial no crossterms no log
    #modeltype 2: polynomial crossterms no log
    #modeltype 3: ridge regression
    #modeltype 4: kernel ridge regression
    #modeltype 5: gaussian process regression
    #modeltype 6: polynomial no crossterms log
    #modeltype 7: polynomial crossterms log
    
    #crossvaltype 1: slicewise, in order k fold
    #crossvaltype 2: random blocks
    #crossvaltype 3: random pixels, no buffer
    #crossvaltype 4: grid blocks, k fold
    #crossvaltype 5: slicewise, random k fold
    
    #errortype 1: mean absolute error
    #errortype 2: mean error
    #errortype 3: RMS error
    
    #datanum 1: Zhongnan v0 data
    #datanum 2: Zhongnan v1 data
    #datanum 3: Jie Pt data
    #datanum 4: Jie PtMo data
    
    #Runs all model variations and adds results to a dataframe. Writes result 
    #to a CSV file.
    for datanum in datanumber:   
        if datanum == 1:
            datafile1 = loadmat('Pt110-multislice-v0.mat')['ImgG']
            datafile2 = np.load('Pt110-convolution-v0.npy')
            shape = datafile1.shape
            
            ms_data = []
            conv_data = []
            for s in range(shape[2]):
                d1 = datafile1[:,:,s]
                d2 = datafile2[:,:,s]
                
                sliceshape = d1.shape
                for i in range(sliceshape[0]):
                    for j in range(sliceshape[1]):
                        if d2[i,j] < 1e-8:
                            d2[i,j] = 1e-8                
                ms_data.append(d1)
                conv_data.append(d2)
    #        ms_data = datafile1['ImgG']
    #        conv_data = datafile2
        elif datanum == 2:
            datafile1 = np.load('Pt110-multislice-v1.npy')
            datafile2 = np.load('Pt110-convolution-v1.npy')
            shape = datafile1.shape
            
            ms_data = []
            conv_data = []
            for s in range(shape[2]):
                d1 = datafile1[:,:,s]
                d2 = datafile2[:,:,s]
                
                sliceshape = d1.shape
                for i in range(sliceshape[0]):
                    for j in range(sliceshape[1]):
                        if d2[i,j] < 1e-8:
                            d2[i,j] = 1e-8                
                ms_data.append(d1)
                conv_data.append(d2)
                
        elif datanum == 3:
            ms_data = []
            conv_data = []     
            for i in range(0,20):
                convname = 'Pt_convolution/Pt_convolution_' + str(i) + '.txt'
                msname = 'Pt_multislice_16_phonons/Pt_' + str(i) + '_cl160mm_ss.tif'
                conv = np.loadtxt(convname)
                ms = np.array(Image.open(msname), dtype = 'float64')
                
                sliceshape = conv.shape
                for i in range(sliceshape[0]):
                    for j in range(sliceshape[1]):
                        if conv[i,j] < 1e-8:
                            conv[i,j] = 1e-8
                ms_data.append(ms)
                conv_data.append(conv)
                
        elif datanum == 4:
            ms_data = []
            conv_data = []     
            for i in range(0,20):
                convname = 'Pt-Mo_convolution/Pt_Mo_convolution_' + str(i) + '.txt'
                msname = 'Pt-Mo_multislice/pt_mo_' + str(i) + '_cl160mm_ss.tif'
                conv = np.loadtxt(convname)
                ms = np.array(Image.open(msname), dtype = 'float64')
                
                sliceshape = conv.shape
                for i in range(sliceshape[0]):
                    for j in range(sliceshape[1]):
                        if conv[i,j] < 1e-8:
                            conv[i,j] = 1e-8
                ms_data.append(ms)
                conv_data.append(conv)
    
        pctmaerror = []
        pctmerror = []
        pctrms = []
        maerror = []
        merror = []
        rms = []
        
        c = ['number','fractionin','testcen','traincen','outnum','alltestnum','degree','inputs','terms', 
        'blocksize','valtype','modeltype','timeperpixel','timesd','maepct','mepct','rmspct', 
        'maepctsem','mepctsem','mae','me','rms','maesem','mesem', 'pctimp']
        
        result = pd.DataFrame(columns = c)
        
        currrow = 0
        
        for fi in fracin:
            for d in deg:
                for inp in inputs:
                    for cvt in crossvaltype:
                        for blk in blocksize:
                            for mt in modeltype:
                                for r in repeats:
                                    if blurring in (True, 'True'):
                                        conv_data = buildLIST.blur(conv_data)
                                    terms = buildLIST.findtermnums(d, inp, mt)
                                    pctmaerror = []
                                    pctmerror = []
                                    pctrms = []
                                    maerror = []
                                    merror = []
                                    rms = []
#                                    pctmaemean = []
#                                    pctmemean = []
#                                    pctrmsmean = []
#                                    maemean = []
#                                    memean = []
#                                    rmsmean = []
                                    origerror = []
                                    allpredicted = []
                                    allmstest = []
                                    allxtest = []
                                    allmerror = []
                                    allmaerror = []
                                    allnum = 0
                                    timevec = []
                                        
                                    for rep in range(r): 
                                        modellist, xtestlist, mstestlist, slinflist, traincen, testcen, outnum = buildLIST.modelkfold(ms_data, 
                                                conv_data, d, inp, mt, cvt, fi, blk)
                                        i = 0
#                                        pctmaecurr = []
#                                        pctmecurr = []
#                                        pctrmscurr = []
#                                        maecurr = []
#                                        mecurr = []
#                                        rmscurr = []
                                        k = 0
                                        l = 0
                                        for model in modellist:
                                            X_test = xtestlist[i]
                                            ms_test = mstestlist[i]
                                        
                                            predicted, time = traintestLIST.predict(X_test, d, inp, 
                                                                              model, mt)
                                            
                                            allpredicted.extend(predicted)
                                            allmstest.extend(ms_test)
                                            allxtest.extend(X_test)
                                
                                            allnum = allnum + len(predicted)
                                            
                                            timevec.append(time/len(predicted))
                                            
                                            #CREATE PLOT(S) HERE
                                            #if slicewise cross val, create difference plots
                                            if (cvt == 5 or cvt == 1) and save == True:
                                                slicesinfolds = slinflist[i]
                                                inorderslices = []
                                                shapes = []
                                                totalslices = len(ms_data)
                                                for s in range(totalslices):
                                                    if s in slicesinfolds:
                                                        shapes.append(ms_data[s].shape)
                                                        inorderslices.append(s)
                
                                                difimagelist, actualimagelist = analyzeLIST.getimages(predicted, 
                                                                                                      ms_test, 
                                                                                                      slicesinfolds, 
                                                                                                      shapes, 
                                                                                                      inputs)
                                                for j in range(len(slicesinfolds)):
                                                    k = k+1
                                                    difimage = difimagelist[j]
                                                    actualimage = actualimagelist[j]
                                                    analyzeLIST.difzoomcompare(difimage, actualimage, 
                                                                                inorderslices[j], 
                                                                                rep=rep, fold=k, 
                                                                                number=number,
                                                                                model=mt, frac=1-fi, 
                                                                                dn=datanum,
                                                                                deg=d, inputs=inp,
                                                                                showzoom=True, 
                                                                                shownozoom=True, 
                                                                                save=save)
                                        
                                            #Create parity plots (for all cross val types)--for all test data
                                            l = l+1
                                            analyzeLIST.allparity(predicted, ms_test, 
                                                                  rep=rep, fold=l, 
                                                                  number=number, 
                                                                  model=mt, frac=1-fi, 
                                                                  dn=datanum, deg=d, 
                                                                  inputs=inp, save=save)
                                            
#                                            timevec.append(time)
#                                            origerror.append(analyzeLIST.geterror(3, False, X_test, ms_test))
#                                            pct = True
#                                            pctmaecurr.append(analyzeLIST.geterror(1, pct, predicted, ms_test))
#                                            pctmecurr.append(analyzeLIST.geterror(2, pct, predicted, ms_test))
#                                            pctrmscurr.append(analyzeLIST.geterror(3, pct, predicted, ms_test))
#                                            pct = False
#                                            maconvval = analyzeLIST.geterror(1, pct, predicted, ms_test)
#                                            mconvval = analyzeLIST.geterror(2, pct, predicted, ms_test)
#                                            rmsconvval = analyzeLIST.geterror(3, pct, predicted, ms_test)
#                                            maecurr.append(maconvval)
#                                            mecurr.append(mconvval)
#                                            rmscurr.append(rmsconvval)
                                            i = i + 1 
                                        
#                                        pctmaemean.append(np.mean(pctmaecurr))
#                                        pctmemean.append(np.mean(pctmecurr))
#                                        pctrmsmean.append(np.mean(rmscurr))
#                                        maemean.append(np.mean(maecurr))
#                                        memean.append(np.mean(mecurr))
#                                        rmsmean.append(np.mean(rmscurr))
#                                        
#                                        pctmaerror.append(pctmaecurr)
#                                        pctmerror.append(pctmecurr)
#                                        pctrms.append(pctrmscurr)
#                                        maerror.append(maecurr)
#                                        merror.append(mecurr)
#                                        rms.append(rmscurr)
                                        
#                                        root = np.sqrt(r)
                                    
                                    for n in range(allnum):
                                        allmerror.append(allpredicted[n]-allmstest[n])
                                        allmaerror.append(abs(allpredicted[n]-allmstest[n]))
#                                    allmerror = np.array(allpredicted) - np.array(allmstest)
#                                    allmaerror = abs(allmerror)
                                    stdmerror = np.std(allmerror)/np.sqrt(allnum)
                                    stdmaerror = np.std(allmaerror)/np.sqrt(allnum)
                                    stdpctmerror = np.std(allmerror*100/np.mean(allmstest))/np.sqrt(allnum)
                                    stdpctmaerror = np.std(allmaerror*100/np.mean(allmstest))/np.sqrt(allnum)
                                    origerror = analyzeLIST.geterror(3, False, allxtest, allmstest)
                                    rms = analyzeLIST.geterror(3, False, allpredicted, allmstest)
                                    maerror = analyzeLIST.geterror(1, False, allpredicted, allmstest)
                                    merror = analyzeLIST.geterror(2, False, allpredicted, allmstest)
                                    pctrms = analyzeLIST.geterror(3, True, allpredicted, allmstest)
                                    pctmaerror = analyzeLIST.geterror(1, True, allpredicted, allmstest)
                                    pctmerror = analyzeLIST.geterror(2, True, allpredicted, allmstest)
                                    
                                    thisrow = np.array([[number, fi, testcen, traincen, outnum, allnum, 
                                                         d, inp, terms, blk, cvt,
                                                         mt, np.mean(timevec), np.std(timevec), 
                                                        np.mean(pctmaerror), np.mean(pctmerror), 
                                                        np.mean(pctrms), stdpctmaerror, 
                                                        stdpctmerror, 
                                                        np.mean(maerror), np.mean(merror), 
                                                        np.mean(rms), stdmaerror, 
                                                        stdmerror, 1-rms/origerror]])
                                    thisrowdf = pd.DataFrame(data = thisrow, columns = c)
                                    result = result.append(thisrowdf, ignore_index = True)
            
                                    currrow = currrow + 1
                    
                    if save in (True, 'True'):
                        date = datetime.now()
    #                    folderdate = date.strftime('%Y-%m-%d_data')
    #                    analyzeLIST.mkdir_p(folderdate)
                        filedate = date.strftime(name+'_'+str(number)+'_%Y-%m-%d_%H;%M;%S')
                        result.to_csv(filedate + '.csv', header = False, index = False)
    #have not updated plots to plot error histograms of model error/conv error fraction                        
    #                        analyze.errorhist(maerror, 'auto', min(maerror), max(maerror), False, 
    #                                            modeltype, crossvaltype, 1, 
    #                                            fracin, deg, inputs)
    #                        analyze.errorhist(merror, 'auto', min(merror), max(merror), False, 
    #                                            modeltype, crossvaltype, 2, 
    #                                            fracin, deg, inputs)
    #                        analyze.errorhist(rms, 'auto', min(rms), max(rms), False, 
    #                                            modeltype, crossvaltype, 3, 
    #                                            fracin, deg, inputs)
    
    #date = datetime.now()
    #name = 'polyblur'
    #folderdate = date.strftime('%Y-%m-%d_data')
    #stemplots.mkdir_p(folderdate)
    #filedate = date.strftime('%Y-%m-%d_%H%M_' + name)
    #
    #result.to_csv(folderdate + '/' + filedate + '.csv', header = True, index = False)
    
    #analyze.errorhist(error, 'auto', min(error), max(error), True, modeltype, crossvaltype, 
    #                    errortype, fractionout, deg, inputs, alpha)
    #print('Cross val type: ' + str(crossvaltype) + '\nModeltype: ' + str(modeltype) + 
    #'\nInputs: ' + str(inputs) + '\nDegree: ' + str(deg))
    #print('Average percent mean absolute error (type ' + str(errortype) + '): ' + str(np.mean(error)))
    #print('Standard deviation: ' + str(np.std(error)))
    return result