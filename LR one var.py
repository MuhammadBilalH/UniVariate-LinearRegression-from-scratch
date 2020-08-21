# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 10:25:10 2020

@author: FADED
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

############################################################
##FUN
def costFun(X_norm,Y_norm,theta):
    lr=np.matmul(X_norm, theta)
    x=(lr.flatten()-Y_norm.flatten())
    J =  (x** 2)/2/m;
    return J

###########################################################




data=pd.read_csv('ex1data1.txt') #read data from .txt file
X=data.iloc[:,0:1]
Y=pd.DataFrame(data['dep'])

m=len(Y) #len of data

#normalizing variables
X_norm=pd.DataFrame((X['ind']-X['ind'].min())/X['ind'].mean())
X_norm.insert(0,column='ones',value=np.ones(m))
X_norm=X_norm.iloc[:,:].values
Y_norm=pd.DataFrame((Y['dep']-Y['dep'].mean())/Y['dep'].std()).values

#visul=plt.scatter(data['ind'],data['dep'],color='red')
#plt.show()          #visualize data

#initialize params
theta=np.zeros((2,1))
itr=2000
alpha=0.1
J_history=np.zeros((m,itr))

#train data
for i in range(itr):
    theta[0] = theta[0] - np.dot(X_norm[:,0],(np.matmul(X_norm, theta) - Y_norm)*0.03);
    theta[1] = theta[1] - np.dot(X_norm[:,1],(np.matmul(X_norm, theta) - Y_norm)*0.03);
    J_history[:,i] = costFun(X_norm, Y_norm, theta); #save cost history
    
    if (i % 200 == 0): #print values of theta every 200 iterations
        print('Theta found by gradient descent: ')
        print( theta[0], theta[1])
    

# print final values of theta to screen
print('Theta found by gradient descent: ')
print( theta[0], theta[1])

#plot linear regression on scater plot of data
plt.plot(X_norm[:,1], np.matmul(X_norm,theta) ,color='Red')
pltLR=plt.scatter(X_norm[:,1],Y_norm,color='Green')
plt.show()
