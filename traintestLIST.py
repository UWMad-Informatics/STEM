#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 12:39:28 2017

@author: aidan
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn import linear_model
import statsmodels.formula.api as smf
import pandas as pd
import patsy
import buildLIST
from timeit import default_timer as timer  
import math

#Fits a polynomial model of specified degree, number of inputs, and presence of 
#cross terms.
def poly(X_train, ms_train, deg, inputs, cross, logterm):
    if cross == False:
        names, completenames, formula = getformula(deg, inputs, False, cross, logterm)
        
        if logterm == True:
            X_train = addlog(X_train)
        df = pd.DataFrame(data=np.column_stack((X_train, ms_train)), 
                          columns = completenames)
        polyfit = smf.ols(formula, df).fit()

    if cross==True:
        #fits model using cross terms
        if logterm == True:
            X_train = addlog(X_train)
        poly = PolynomialFeatures(degree=deg)
        X_train_transform = poly.fit_transform(X_train)
    
        #fits model using data from transformed vector
        polyfit = linear_model.LinearRegression()
        polyfit.fit(X_train_transform, ms_train)
        
    return polyfit

#Fits a polynomial model of specified degree, number of inputs, and presence of 
#cross terms using the R wrapper. Does no cross terms and some cross terms.
def polysmf(X_train, ms_train, deg, inputs, cross, logterm):
    names, completenames, formula = getformula(deg, inputs, False, cross, logterm) 
    df = pd.DataFrame(data=np.column_stack((X_train, ms_train)), 
                      columns = completenames)
    polyfit = smf.ols(formula, df).fit()   
    return polyfit

#Fits ridge regression model
def rr(X_train, ms_train):
    rrfit = Ridge(alpha = 1)
    rrfit.fit(X_train,ms_train)    
    return rrfit

#Fits kernel ridge regression model
def krr(X_train, ms_train):
    krrfit = KernelRidge(alpha = 1)
    krrfit.fit(X_train,ms_train)   
    return krrfit

#Fits gaussian process regression model
def gpr(X_train, ms_train):
    gprfit = GaussianProcessRegressor(kernel = None)
    gprfit.fit(X = X_train, y = ms_train)
    return gprfit

#Takes a model and a test set and gets a list of predicted values
def predict(X_test, deg, inputs, model, modeltype):
    if modeltype == 1 or modeltype == 6:
        cross = False
        if modeltype == 1:
            logterm = False
        if modeltype == 6:
            logterm = True
            X_test = addlog(X_test)
        names, completenames, formula = getformula(deg, inputs, True, cross, logterm)
        df = pd.DataFrame(data=X_test, columns = names)
        x = patsy.dmatrix(formula, data=df)
        start = timer()
        predicted = model.predict(x, transform=False)
        end = timer()
        time = (end - start)/len(predicted)   
        
    if modeltype == 2 or modeltype == 7:
        poly = PolynomialFeatures(degree = deg)
        if modeltype == 7:
            X_test = addlog(X_test)
        X_test_transform = poly.fit_transform(X_test)
        start = timer()
        predicted = model.predict(X_test_transform)
        end = timer()
        time = (end - start)/len(predicted)
        
    elif modeltype == 3 or modeltype == 4 or modeltype == 5:
        start = timer()
        predicted = model.predict(X_test)
        end= timer()
        time = (end - start)/len(predicted)
        
    return predicted, time

#constructs the correct formula based on degree, number of inputs, and polynomial type
def getformula(deg, inputs, predict, cross, logterm):
    names = []
    completenames = []
    center = math.floor(inputs/2)
    for n in range(inputs):
        names.append('p'+str(n))
        completenames.append('p'+str(n))
    if logterm == True:
        names.append('np.log(p' + str(center) + ')')
        completenames.append('np.log(p' + str(center) + ')')
    completenames.append('ms')
    
    if predict == False:
        formula = 'ms ~ '
    else:
        formula = ''
        
    count = 0
    for n in names:
        for d in range(deg):
            if count != 0:
                formula = formula + ' + np.power('+str(n)+', '+str(d+1)+')'
            else:
                formula = formula + 'np.power('+str(n)+', '+str(d+1)+')'
            count = count + 1
            
    return names, completenames, formula

#adds a column into x that represents the log of the center value
def addlog(x):
    inputs = x.shape[1]
    center = math.floor(inputs/2)
    newx = np.zeros([x.shape[0], x.shape[1]+1])
    newx[:,:-1] = x
    newx[:,-1:] = np.reshape(np.log10(x[:,center]), [x.shape[0], 1])
    return newx