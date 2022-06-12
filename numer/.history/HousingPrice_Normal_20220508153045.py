# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:22:07 2021

@author: Vik Gupta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HP = pd.read_excel (r'new apartment prices in Longgang District.xlsx')
HP = pd.DataFrame(HP, columns= ['SIZE(M2)','PRICE(10000RMB)'])
HP = HP.to_numpy()

# number of examples in the training set and features + label 
m, n = HP.shape

X = HP[:,0].reshape(n-1,m)
Y = HP[:,1].reshape(1,m)

# Change array type to float
X = X.astype('float64')
Y = Y.astype('float64')

# normalize X
X = X-0.5*np.max(X)-0.5*np.min(X)

X = X/np.sqrt(np.mean(X**2))

# Add the dummy feature 1 to X
X = np.concatenate((np.ones((1,m),dtype=float),X),axis=0)

w = np.dot(np.linalg.inv(np.dot(X,np.transpose(X))),np.dot(X,
           np.transpose(Y)))

Ypred = np.dot(np.transpose(w),X)
    
J  = 0.5*np.sum((Ypred - Y)**2)

fig4, ax4 = plt.subplots()
ax4.scatter(X[1,:],Y)
ax4.plot(X[1,:],Ypred.reshape(61,))
