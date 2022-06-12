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

def Normalization(x):
    xmax = max(x[0,:])
    xmin = min(x[0,:])
    for i,val in enumerate(x[0,:]):
        # print(val)
        x[0,i]=float(val-xmin)/float(xmax-xmin)
        # print(x[0,i])
    print(min(x[0,:]))
# print(Y)
Normalization(X)
Normalization(Y)

# Add the dummy feature 1 to X
X = np.concatenate((np.ones((1,m),dtype=float),X,X**0.5),axis=0)
# 添加了一列为x的开方
n = X.shape[0]

# Initialize the weights
w = np.zeros((n,1),dtype=float)

# The learning parameter in linear regression is the leraning rate
# alpha = 0.0000055  # This parameter should be optimized
alpha = 0.000001  # This parameter should be optimized

Imax  = 2000      # Maximum number of iterations

# Run the batch gradient descent algorithm

J  = np.zeros((1,Imax),dtype=float)
dJ = np.zeros((n,Imax),dtype=float)

fig5, ax5 = plt.subplots()
ax5.scatter(X[1,:],Y)
ax5.set_title("The iteration result(every 100 times)"+"when $\\alpha$="+str(alpha))

for t in range(2000):
    
    Ypred = np.dot(np.transpose(w),X)
    
    J[0,t]  = 0.5*np.sum((Ypred - Y)**2)
    
    dJ[:,t:t+1] = np.mean((Ypred-Y)*X,axis=1).reshape(n,1)
    
    w -= alpha*dJ[:,t:t+1]
    # if t%100 == 0:
    #     ax5.plot(X[1,:],Ypred.reshape(61,))

J = J.reshape(Imax,1)
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(len(J)),J)
# ax1.set_title("The $J_{min}$ result is "+str(J[-1])+"$\\alpha$="+str(alpha))


# fig2, ax2 = plt.subplots()
# ax2.plot(np.arange(len(J)),dJ[0,:])
# ax2.set_title("The $d(J)[0,:]$ result"+"$\\alpha$="+str(alpha))


# fig3, ax3 = plt.subplots()
# ax3.plot(np.arange(len(J)),dJ[1,:])
# ax3.set_title("The dJ[1,:] result"+"$\\alpha$="+str(alpha))


# fig4, ax4 = plt.subplots()
# ax4.plot(np.arange(len(J)),dJ[2,:])
# ax4.set_title("The dJ[2,:] result"+"$\\alpha$="+str(alpha))


plt.show()