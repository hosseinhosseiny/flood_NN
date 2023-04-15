# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:54:26 2019

@author: shossein
"""


import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import os
cwd=os.getcwd()
from sklearn.model_selection import train_test_split
#wd=os.chdir("hosseinhosseiny") 
  
data= pd.read_csv("results.csv")
X_h = data[:][['X','Y','Q_cms']]
y_h = data[:]['Depth']
X_wd=data[:][['X','Y','Q_cms']]
y_wd = data[:]['IBC']
X_z = data[:][['X','Y']]
y_z = data[:]['Elevation']

# test Q600
#----------------------- test for Q=600 
#
data600= pd.read_csv("Result_Q300_mesh1m.csv")
X_h600 = data600[:][['X','Y','Q']]
y_h600 = data600[:]['Depth']
X_wd600=data600[:][['X','Y','Q']]
y_wd600 = data600[:]['IBC']
X_z600= data600[:][['X','Y']]
y_z600= data600[:]['Elevation']

exp300= pd.read_csv("expo300_z_h.csv")
#exp300.columns = ["z", "h"]

#---------- Splitting data for train and test
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.3, random_state=0)
X_train_wd, X_test_wd, y_train_wd, y_test_wd = train_test_split(X_wd, y_wd, test_size=0.3, random_state=5)
X_train_z, X_test_z, y_train_z, y_test_z = train_test_split(X_z, y_z, test_size=0.3, random_state=2)

import time

time_start = time.perf_counter()


from sklearn.preprocessing import StandardScaler
from sklearn import metrics
sc = StandardScaler()
## ---------Scaling the data input for he estimate

#X_train_h_sc = sc.fit_transform(X_train_h)
#X_h_sc600 = sc.fit_transform(X_h600)
#
#X_test_h_sc = sc.transform(X_test_h)
#
#
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
#
#print('#--------------------Ranodm forest regression for h\n')
#      
#regressor_h = RandomForestRegressor(n_estimators=10, random_state=0,n_jobs=-1)
#
#regressor_h.fit(X_train_h, y_train_h)
#y_pred_h = regressor_h.predict(X_test_h)
#y_pred_h600 = regressor_h.predict(X_h600)
#
#time_elapsed = (time.perf_counter() - time_start)

#print ("%5.1f secs " % (time_elapsed))

#Evaluating h regression error
#print('Mean Absolute Error regression depth:', metrics.mean_absolute_error(y_test_h, y_pred_h))
#print('Mean Squared Error regression depth:', metrics.mean_squared_error(y_test_h, y_pred_h))
#print('Root Mean Squared Error regression depth:', np.sqrt(metrics.mean_squared_error(y_test_h, y_pred_h)))
#
#print('results for Q=600\n')
#
#print('Mean Absolute Error regression depth for q=600:', metrics.mean_absolute_error(y_h600, y_pred_h600))
#print('Mean Squared Error regression depth:', metrics.mean_squared_error(y_h600, y_pred_h600))
#print('Root Mean Squared Error regression depth:', np.sqrt(metrics.mean_squared_error(y_h600, y_pred_h600)))
#print('\n')
#



print('#-------------------Random forest for classification for wet dry\n')
      
X_train_wd_sc = sc.fit_transform(X_train_wd)
X_test_wd_sc = sc.transform(X_test_wd)
X_wd_sc600 = sc.transform(X_wd600)


clf= RandomForestClassifier(n_estimators=20, random_state=5,n_jobs=-1)
clf.fit(X_train_wd_sc, y_train_wd)
y_pred_wd = clf.predict(X_test_wd_sc)
y_pred_wd600 = clf.predict(X_wd_sc600)
exp300["wd_rf"]=y_pred_wd600
exp300["wd_iric"]=data600['IBC']
exp300["wse_ann"]=0


c_exp300 = exp300.copy()
c_exp300['h_ann'][c_exp300['h_ann']< 0] = 0
c_exp300['h_ann'][c_exp300['wd_rf']==0] = 0
for i in range (len(c_exp300['h_ann'])):
    if c_exp300['wd_rf'][i]!=0:
        c_exp300['wse_ann'][i]= c_exp300['h_ann'][i]+c_exp300['z_ann'][i]
        
c_exp300.to_csv("output300.csv")

#print('Mean Absolute Error classify wet dry:', metrics.mean_absolute_error(y_wd600, y_pred_wd600))
#print('Mean Squared Error classify wet dry:', metrics.mean_squared_error(y_test_wd, y_pred_wd))
#print('Root Mean Squared Error classify wet dry:', np.sqrt(metrics.mean_squared_error(y_test_wd, y_pred_wd)))
#print('\n')

#print('#--confusion matrix for wet or dry\n')

#from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#
#print(confusion_matrix(y_test_wd,y_pred_wd))
#print(classification_report(y_test_wd,y_pred_wd))
#print(accuracy_score(y_test_wd, y_pred_wd))
#print('\n')
##
#print('#------------------Ranodm forest regression for z\n')
#      
#
#X_train_z_sc = sc.fit_transform(X_train_z)
#X_test_z_sc = sc.transform(X_test_z)
#
#regressor_z = RandomForestRegressor(n_estimators=10, random_state=10,n_jobs=-1)
#regressor_z.fit(X_train_z_sc, y_train_z)
#y_pred_z = regressor_z.predict(X_test_z_sc)
##Evaluating h regression error
#print('Mean Absolute Error regression elevation:', metrics.mean_absolute_error(y_test_z, y_pred_z))
#print('Mean Squared Error regression elevation:', metrics.mean_squared_error(y_test_z, y_pred_z))
#print('Root Mean Squared Error regression elevation:', np.sqrt(metrics.mean_squared_error(y_test_z, y_pred_z)))
#
##-------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------
#
##
###
##X_train_h600, X_test_h600, y_train_h600, y_test_h600 = train_test_split(X_h600_in, y_h600_in, test_size=1, random_state=0)
##X_train_wd600, X_test_wd600, y_train_wd600, y_test_wd600 = train_test_split(X_wd600_in, y_wd600_in, test_size=0, random_state=5)
##X_train_z600, X_test_z600, y_train_z600, y_test_z600 = train_test_split(X_z600_in, y_z600_in, test_size=0, random_state=2)
#
#
##      
##X_wd_sc600 = sc.transform(X_test_wd600)
##y_pred_wd600 = clf.predict(X_wd_sc600)
##print('Mean Absolute Error classify wet dry:', metrics.mean_absolute_error(y_wd600_in, y_pred_wd600))
##print('Mean Squared Error classify wet dry:', metrics.mean_squared_error(y_wd600_in, y_pred_wd600))
##print('Root Mean Squared Error classify wet dry:', np.sqrt(metrics.mean_squared_error(y_wd600_in, y_pred_wd600)))
##print('\n')
##
##print('-------------------Random forest elevation prediction for Q=600\n')
##
##
##X_z_sc600 = sc.transform(X_test_z600)
#y_pred_z600 = regressor_z.predict(X_z_sc600)
##Evaluating h regression error
#print('Mean Absolute Error regression elevation:', metrics.mean_absolute_error(y_z600_in, y_pred_z600))
#print('Mean Squared Error regression elevation:', metrics.mean_squared_error(y_z600_in, y_pred_z600))
#print('Root Mean Squared Error regression elevation:', np.sqrt(metrics.mean_squared_error(y_z600_in, y_pred_z600)))
#
#
#
#
#
#
#
