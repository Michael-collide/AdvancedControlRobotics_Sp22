# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:26:10 2021
@author: Vik Gupta

Python is an open-source high-level general purpose programming language.

Open-source - Means it is freely available with all its tools, like C, C++
              Matlab is not free, you need to pay for additional tools, it
              you can be banned from using Matlab if the company wants, many
              Chinese universities are banned from obtaining new licence for matlab
              
High-level - It is an interpreted language. It is at higher-level than C and C++,
             which means it is easier to use but is also slower. Many underlying
             algorithms in Python work on C++

General-purpose - It is an extremely powerful language and can be used for scientific
                  calculations, machine learning, game development, web development, etc.

It manages all these applications by having numerous libraries built for specific
applications. The key is to import those libraries.

To replace matlab, one can use math, numpy, scipy, matplotlib and panda libraries.
Later in this course, we will import torch library to code neural networks.

Finally, python programming is simple and fun. Let us walk through it for linear regression
problem here to fit housing prices with their size.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HP = pd.read_excel (r'new apartment prices in Longgang District.xlsx')
HP = pd.DataFrame(HP, columns= ['SIZE(M2)','PRICE(10000RMB)'])
print (HP)

HP = HP.to_numpy()
print(HP.shape)

# number of examples in the training set and features + label 
m, n = HP.shape


X = HP[:,0].reshape(n-1,m)
Y = HP[:,1].reshape(1,m)

print(X.shape)
print(X.dtype)

# Change array type to float
X = X.astype('float64')
Y = Y.astype('float64')
print(X.dtype)

# normalize X - Try to run the code without this normalization and
# compare the final cost function with the normal form equation results
X = X-0.5*np.max(X)-0.5*np.min(X)

X = X/np.sqrt(np.mean(X**2))

# Add the dummy feature 1 to X
X = np.concatenate((np.ones((1,m),dtype=float),X),axis=0)
print(X.shape)

# Initialize the weights
w = np.zeros((n,1),dtype=float)

# The learning parameter in linear regression is the leraning rate
alpha = 1.28  # This parameter should be optimized
Imax  = 491      # Maximum number of iterations

# Run the batch gradient descent algorithm

J  = np.zeros((1,Imax),dtype=float)
dJ = np.zeros((n,Imax),dtype=float)

fig4, ax4 = plt.subplots()
ax4.scatter(X[1,:],Y)

for t in range(Imax):
    
    Ypred = np.dot(np.transpose(w),X)
    
    J[0,t]  = 0.5*np.sum((Ypred - Y)**2)
    
    dJ[:,t:t+1] = np.mean((Ypred-Y)*X,axis=1).reshape(n,1)
    
    w -= alpha*dJ[:,t:t+1]
    
    if t%49 == 0:
        ax4.plot(X[1,:],Ypred.reshape(61,))
#        input("Press Enter to continue...")

plt.show

J = J.reshape(Imax,1)

fig1, ax1 = plt.subplots()
ax1.plot(np.arange(len(J)),np.log(J))

fig2, ax2 = plt.subplots()
ax2.plot(np.arange(len(J)),dJ[0,:])

fig3, ax3 = plt.subplots()
ax3.plot(np.arange(len(J)),dJ[1,:])



















