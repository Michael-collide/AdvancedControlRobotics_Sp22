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
print ("HP is :",HP)

HP = HP.to_numpy(copy=True)

# number of examples in the training set and features + label 
m, n = HP.shape


X = HP[:,0].reshape(n-1,m)
Y = HP[:,1].reshape(1,m)

# print(X.shape)
# print(X.dtype)

# Change array type to float
X = X.astype('float64')
Y = Y.astype('float64')
# print(X.dtype)

# Add the dummy feature 1 to X
X = np.concatenate((np.ones((1,m),dtype=float),X),axis=0)
# print(X.shape)

# Initialize the weights
w = np.zeros((n,1),dtype=float)

# The learning parameter in linear regression is the leraning rate
# alpha = 0.000001  # This parameter should be optimized
alpha = 0.0002005  # This parameter should be optimized

Imax  = 2000      # Maximum number of iterations

# Run the batch gradient descent algorithm

J  = np.zeros((1,Imax),dtype=float)
dJ = np.zeros((n,Imax),dtype=float)

fig4, ax4 = plt.subplots()
ax4.scatter(X[1,:],Y)
# ax4.set_title("The iteration result(every 100 times)"+"when $\\alpha$="+str(alpha))

for t in range(2000):
    
    Ypred = np.dot(np.transpose(w),X)
    
    J[0,t]  = 0.5*np.sum((Ypred - Y)**2)
    
    dJ[:,t:t+1] = np.mean((Ypred-Y)*X,axis=1).reshape(n,1)
    
    w -= alpha*dJ[:,t:t+1]
    
    # if t%100 == 0:
        # ax4.plot(X[1,:],Ypred.reshape(61,))

J = J.reshape(Imax,1)
print(w)
fig1, ax1 = plt.subplots()
ax1.plot(np.arange(len(J)),J)
# ax1.set_title("The $J_{min}$ result is (batch gradient descent) "+str(J[-1]))
ax1.legend(["Learning rate is $\\alpha$="+str(alpha)], loc = 'upper right')
ax4.plot(X[1,:],Ypred.reshape(61,))
ax4.set_title("Using batch-gradient-descent")
fig2, ax2 = plt.subplots()
ax2.plot(np.arange(len(J)),dJ[0,:])
ax2.set_title("The $d(J)[0,:]$ result"+"$\\alpha$="+str(alpha))

fig3, ax3 = plt.subplots()
ax3.plot(np.arange(len(J)),dJ[1,:])
ax3.set_title("The dJ[1,:] result"+"$\\alpha$="+str(alpha))
plt.show()
print(J[-1])










