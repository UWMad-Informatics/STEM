#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 15:40:48 2017

@author: aidan
"""
import numpy as np
import pandas as pd

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
    
c = ['rownum','datanum','fractionin','degree','inputs','valtype','blocksize',
     'modeltype','repeats','name','blurring','save']

name = 'Pt_PtMo_run_0'
datanum = [3, 4]
fracin = [.9, .8, .7, .6, .5]
deg = [1, 2, 3]
inputs = [1, 9, 25]
crossvaltype = [3, 4, 5]
blocksize = [7]
modeltype = [1, 2, 6, 7]
repeats = [2]
blurring = [False]
save = True
totalnum = len(datanum)*len(fracin)*len(deg)*len(inputs)*len(blocksize)*len(crossvaltype)*len(modeltype)*len(repeats)*len(blurring)
rownum = 0
smalllist = []
#medlist = [52, 54, 60, 62, 65, 67, 68, 70, 88, 98, 106, 136, 212, 280, 282, 316, 318, 426, 532, 534]
#largelist = [28, 90, 96, 104, 172, 174, 208, 210, 279, 302, 303, 307, 321, 421, 423, 425, 427, 429, 431, 529, 531, 533, 535, 537, 539]
#xlargelist = [30, 64, 66, 97, 99, 100, 101, 102, 103, 105, 107, 205, 207, 209, 211, 213, 215, 294, 313, 315, 317, 319, 323] 
medlist = []
largelist = []
xlargelist = []

medaddon = np.ones_like(medlist)*int(totalnum/len(datanum)) + medlist
largeaddon = np.ones_like(largelist)*int(totalnum/len(datanum)) + largelist
xlargeaddon = np.ones_like(xlargelist)*int(totalnum/len(datanum)) + xlargelist

medlist.extend(medaddon)
largelist.extend(largeaddon)
xlargelist.extend(xlargeaddon)
   
result = pd.DataFrame()

for data in datanum:
    for fi in fracin:
        for d in deg:
            for inp in inputs:
                for cvt in crossvaltype:
                    for blk in blocksize:
                        for mt in modeltype:
                            for r in repeats:
                                for b in blurring:
                                    thisrow = np.array([[rownum, data, fi, d, 
                                                         inp, cvt, blk,
                                                         mt, r, name, b, save]])
                                    rownum = rownum + 1                                
                                    thisrowdf = pd.DataFrame(data = thisrow, columns = c)
                                    result = result.append(thisrowdf, ignore_index = True)
                                    
result_small = pd.DataFrame()
result_med = pd.DataFrame()
result_large = pd.DataFrame()
result_xlarge = pd.DataFrame()


for i in range(totalnum):
    if i not in medlist and i not in largelist and i not in xlargelist:
        smalllist.append(i)
        
for i in smalllist:
    result_small = result_small.append(result.iloc[i])
for i in medlist:
    result_med = result_med.append(result.iloc[i])
for i in largelist:
    result_large = result_large.append(result.iloc[i])
for i in xlargelist:
    result_xlarge = result_xlarge.append(result.iloc[i])
 
result_small.to_csv('entireinputset.csv', header = True, index = False)                       
#result_small.to_csv(name + '_small.csv', header = False, index = False)
#result_med.to_csv(name + '_med.csv', header = False, index = False)
#result_large.to_csv(name + '_large.csv', header = False, index = False)
#result_xlarge.to_csv(name + '_xlarge.csv', header = False, index = False)
