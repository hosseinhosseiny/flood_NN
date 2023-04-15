# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:54:26 2019

@author: shossein
"""


import pandas as pd
import numpy as np
     
data= pd.read_excel("Test_.xlsx").T
data[0].describe().transpose()
x=np.zeros((1,len(data.T)))
y=np.zeros((1,len(data.T)))
q=np.zeros((1,len(data.T)))
for i in range (len(data.T)):
    x[0,i]= data[i][0]
    y[0,i]= data[i][1]
    q[0,i]= data[i][2]

from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(30,30,30), random_state=1)
####################### checking the dimensions of result
mlp.fit(x,y)  
