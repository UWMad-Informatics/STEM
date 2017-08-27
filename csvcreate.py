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
    
#    c = ['datanum','fractionin','degree','inputs','valtype','blocksize',
#         'modeltype','repeats','name','blurring','save']

name = 'PtMorun_4'
datanum = [4]
fracin = [.9, .8, .7, .6, .5]
deg = [1, 2, 3]
inputs = [1, 9, 25]
crossvaltype = [3, 4, 5]
blocksize = [7]
modeltype = [1, 2, 6, 7]
repeats = [2]
blurring = [False]
save = True
    
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
                                    thisrow = np.array([[data, fi, d, 
                                                         inp, cvt, blk,
                                                         mt, r, name, b, save]])
                                    thisrowdf = pd.DataFrame(data = thisrow)
                                    result = result.append(thisrowdf, ignore_index = True)
                        
result.to_csv(name + '.csv', header = False, index = False)
