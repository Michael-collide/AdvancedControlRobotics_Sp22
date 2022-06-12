# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 17:10:46 2021

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

# Add the dummy feature 1 to X , now also add sqrt(X) as a feature
X = np.concatenate((np.ones((1,m),dtype=float),X,X**0.5),axis=0)
n = X.shape[0]

# Initialize the weights
w = np.zeros((n,1),dtype=float)

# The learning parameter in linear regression is the leraning rate
alpha = 0.00004  # This parameter should be optimized
Imax  = 2000      # Maximum number of iterations

# Run the batch gradient descent algorithm

J  = np.zeros((1,Imax),dtype=float)
dJ = np.zeros((n,Imax),dtype=float)

for t in range(100):
    
    Ypred = np.dot(np.transpose(w),X)
    
    J[0,t]  = 0.5*np.sum((Ypred - Y)**2)
    
    dJ[:,t:t+1] = np.mean((Ypred-Y)*X,axis=1).reshape(n,1)
    
    w -= alpha*dJ[:,t:t+1]

J = J.reshape(Imax,1)
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(len(J)),np.log(J))

fig2, ax2 = plt.subplots()
ax2.plot(np.arange(len(J)),dJ[0,:])

fig3, ax3 = plt.subplots()
ax3.plot(np.arange(len(J)),dJ[1,:])

fig4, ax4 = plt.subplots()
ax4.plot(np.arange(len(J)),dJ[2,:])
