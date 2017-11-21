#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 18:04:15 2017

@author: aidan
"""

import stemfitrun
import csv 
import sys

file = open('Ptrun_5.csv')
reader = csv.reader(file)

rownum = int(sys.argv[1])

datanum = []
fracin = []
deg = []
inputs = []
crossvaltype = []
blocksize = []
modeltype = []
repeats = []
blurring = [] 
runnum = []

line = 0
for row in reader:
    if line == rownum:
        runnum.append(int(row[0]))
        datanum.append(int(row[1]))
        fracin.append(float(row[2]))
        deg.append(int(row[3]))
        inputs.append(int(row[4]))
        crossvaltype.append(int(row[5]))
        blocksize.append(int(row[6]))
        modeltype.append(int(row[7]))
        repeats.append(int(row[8]))
        name = row[9]
        blurring.append(row[10])
        save = (row[11] == 'True')
    line = line + 1

result = stemfitrun.stemrun(datanum,fracin,deg,inputs,crossvaltype,
                            blocksize,modeltype,repeats,name,
                            blurring,save,runnum)