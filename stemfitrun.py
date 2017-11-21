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
from sklearn.metrics import r2_score

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
    number = number[0]
    
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
        'maepctsem','mepctsem','mae','me','rms','maesem','mesem', 'pctimp','1-r^2','amorphrms', 
        'mixrms', 'xtalrms', 'amorph1-r^2', 'mix1-r^2', 'xtal1-r^2']
        
        result = pd.DataFrame(columns = c)
        
#        currrow = 0
        
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
                                    origerror = []
                                    allpredicted = []
                                    allmstest = []
                                    allxtest = []
                                    allmerror = []
                                    allmaerror = []
                                    allnum = 0
                                    timevec = []
                                    amorphrms = []
                                    mixedrms = []
                                    xtalrms= []
                                    amorphrsq = []
                                    mixedrsq = []
                                    xtalrsq = []
                                        
                                    for rep in range(r): 
                                        modellist, xtestlist, mstestlist, slinflist, traincen, testcen, outnum = buildLIST.modelkfold(ms_data, 
                                                conv_data, d, inp, mt, cvt, fi, blk)
                                        
                                        i = 0
                                        k = 0
                                        l = 0
                                        sliceerror = []
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
                                            
                                            #If slicewise cross val, create difference plots and 
                                            #find the error statistics for the different particle 
                                            #categories
                                            if (cvt == 5 or cvt == 1) and save == True:
                                                slicesinfolds = slinflist[i]
                                                inorderslices = []
                                                shapes = []
                                                totalslices = len(ms_data)
                                                for s in range(totalslices):
                                                    if s in slicesinfolds:
                                                        shapes.append(ms_data[s].shape)
                                                        inorderslices.append(s)
                                        
                                                difvectorlist, actualvectorlist, predictedvectorlist = analyzeLIST.getslicevectors(predicted, 
                                                                                                      ms_test, 
                                                                                                      slicesinfolds, 
                                                                                                      shapes, 
                                                                                                      inp)
                
                                                difimagelist, actualimagelist, predictedimagelist = analyzeLIST.getimages(predicted, 
                                                                                                      ms_test, 
                                                                                                      slicesinfolds, 
                                                                                                      shapes, 
                                                                                                      inp)
                                                
                                                
                                                for j in range(len(slicesinfolds)):
                                                    k = k+1
                                                    actualvector = actualvectorlist[j]
                                                    predvector = predictedvectorlist[j]
                                                    rmserrorval = analyzeLIST.geterror(3, True, predvector, actualvector)
                                                    rsquareval = 1 - r2_score(actualvector, predvector)
                                                    sliceout = slicesinfolds[j]
                                                    
                                                    if datanum == 3:
                                                        if sliceout in [0,1,2,4,5,7,8]:
                                                            amorphrms.append(rmserrorval)
                                                            amorphrsq.append(rsquareval)
                                                        elif sliceout in [9,10,11,12,13,14]:
                                                            mixedrms.append(rmserrorval)
                                                            mixedrsq.append(rsquareval)
                                                        elif sliceout in [3,6,15,16,17,18,19]:
                                                            xtalrms.append(rmserrorval)
                                                            xtalrsq.append(rsquareval) 
                                                            
                                                    if datanum == 4:
                                                        if sliceout in [0,1,3,4,5,16,19]:
                                                            amorphrms.append(rmserrorval)
                                                            amorphrsq.append(rsquareval)
                                                        elif sliceout in [6,7,8,9,10,11]:
                                                            mixedrms.append(rmserrorval)
                                                            mixedrsq.append(rsquareval)
                                                        elif sliceout in [2,12,13,14,15,17,18]:
                                                            xtalrms.append(rmserrorval)
                                                            xtalrsq.append(rsquareval) 
                                                            
                                                    difimage = difimagelist[j]
                                                    actualimage = actualimagelist[j]
                                                    predimage = predictedimagelist[j]
                                                    analyzeLIST.sliceimages(difimage, actualimage, 
                                                                                predimage,
                                                                                inorderslices[j], 
                                                                                rep=rep, fold=k, 
                                                                                number=number,
                                                                                model=mt, frac=1-fi, 
                                                                                dn=datanum,
                                                                                deg=d, inputs=inp,
                                                                                showpred=True,
                                                                                showzoom=True, 
                                                                                shownozoom=True, 
                                                                                save=save)
                                                    
                                           #If this is not slicewise cross validation, use placeholder
                                           #values for the image type error
                                            elif (cvt!=5 and cvt!=1):
                                                amorphrms.append(-1)
                                                amorphrsq.append(-1)
                                                mixedrms.append(-1)
                                                mixedrsq.append(-1)
                                                xtalrms.append(-1)
                                                xtalrsq.append(-1)
                                                
                                            i = i + 1
                                            
                                #Create parity plots (for all cross val types)--for all test data
                                analyzeLIST.allparity(allpredicted, allmstest, 
                                                      number=number, 
                                                      model=mt, frac=1-fi, 
                                                      dn=datanum, deg=d, 
                                                      inputs=inp, save=save)
                                    
                                for n in range(allnum):
                                    allmerror.append(allpredicted[n]-allmstest[n])
                                    allmaerror.append(abs(allpredicted[n]-allmstest[n]))
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
                                amrms = np.mean(amorphrms)
                                mixrms = np.mean(mixedrms)
                                xrms = np.mean(xtalrms)
                                amrsq = np.mean(amorphrsq)
                                mixrsq = np.mean(mixedrsq)
                                xrsq = np.mean(xtalrsq)
                                r_squared = r2_score(allmstest, allpredicted)
                                
                                thisrow = np.array([[number, fi, testcen, traincen, outnum, allnum, 
                                                     d, inp, terms, blk, cvt,
                                                     mt, np.mean(timevec), np.std(timevec), 
                                                    np.mean(pctmaerror), np.mean(pctmerror), 
                                                    np.mean(pctrms), stdpctmaerror, 
                                                    stdpctmerror, 
                                                    np.mean(maerror), np.mean(merror), 
                                                    np.mean(rms), stdmaerror, 
                                                    stdmerror, 1-rms/origerror, 1-r_squared, 
                                                    amrms, mixrms, xrms, amrsq, mixrsq, xrsq]])
                                thisrowdf = pd.DataFrame(data = thisrow, columns = c)
                                result = result.append(thisrowdf, ignore_index = True)
        
#                                currrow = currrow + 1
                    
        if save in (True, 'True'):
            date = datetime.now()
#                    folderdate = date.strftime('%Y-%m-%d_data')
#                    analyzeLIST.mkdir_p(folderdate)
            filedate = date.strftime(name+'_'+str(number)+'_%Y-%m-%d_%H;%M;%S')
            result.to_csv(filedate + '.csv', header = False, index = False)
    return result