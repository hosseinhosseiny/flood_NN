# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:54:26 2019

@author: shossein
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
cwd=os.getcwd()
from sklearn.model_selection import train_test_split
#wd=os.chdir("hosseinhosseiny") 

# plotting calibration profiles
# 
#fig, ax = plt.subplots() 
#fig.set_size_inches(8,5)  
data= pd.read_csv("results.csv")
X_h = data[:][['X','Y','Q_cms']]
y_h = data[:]['Depth']
X_wd=data[:][['X','Y','Q_cms']]
y_wd = data[:]['IBC']
X_z = data[:][['X','Y']]
y_z = data[:]['Elevation']


X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.3, random_state=0)
X_train_wd, X_test_wd, y_train_wd, y_test_wd = train_test_split(X_wd, y_wd, test_size=0.3, random_state=5)
X_train_z, X_test_z, y_train_z, y_test_z = train_test_split(X_z, y_z, test_size=0.3, random_state=2)

# Scaling the data
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

sc = StandardScaler()
X_train_h_sc = sc.fit_transform(X_train_h)
X_test_h_sc = sc.transform(X_test_h)

X_train_wd_sc = sc.fit_transform(X_train_wd)
X_test_wd_sc = sc.transform(X_test_wd)

X_train_z_sc = sc.fit_transform(X_train_z)
X_test_z_sc = sc.transform(X_test_z)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

print('#--------------------Ranodm forest regression for h\n')
regressor_h = RandomForestRegressor(n_estimators=5, random_state=0)
regressor_h.fit(X_train_h_sc, y_train_h)
y_pred_h = regressor_h.predict(X_test_h_sc)
#Evaluating h regression error
print('Mean Absolute Error regression depth:', metrics.mean_absolute_error(y_test_h, y_pred_h))
print('Mean Squared Error regression depth:', metrics.mean_squared_error(y_test_h, y_pred_h))
print('Root Mean Squared Error regression depth:', np.sqrt(metrics.mean_squared_error(y_test_h, y_pred_h)))
print('\n')
print('#-------------------Random forest for classification for wet dry\n')
clf= RandomForestClassifier(n_estimators=5, random_state=5)
clf.fit(X_train_wd_sc, y_train_wd)
y_pred_wd = clf.predict(X_test_wd_sc)
print('Mean Absolute Error classify wet dry:', metrics.mean_absolute_error(y_test_wd, y_pred_wd))
print('Mean Squared Error classify wet dry:', metrics.mean_squared_error(y_test_wd, y_pred_wd))
print('Root Mean Squared Error classify wet dry:', np.sqrt(metrics.mean_squared_error(y_test_wd, y_pred_wd)))
print('\n')

print('#------------------confusion matrix for wet or dry\n')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test_wd,y_pred_wd))
print(classification_report(y_test_wd,y_pred_wd))
print(accuracy_score(y_test_wd, y_pred_wd))
print('\n')

print('#------------------Ranodm forest regression for z\n')
regressor_z = RandomForestRegressor(n_estimators=1, random_state=10)
regressor_z.fit(X_train_z_sc, y_train_z)
y_pred_z = regressor_z.predict(X_test_z_sc)
#Evaluating h regression error
print('Mean Absolute Error regression elevation:', metrics.mean_absolute_error(y_test_z, y_pred_z))
print('Mean Squared Error regression elevation:', metrics.mean_squared_error(y_test_z, y_pred_z))
print('Root Mean Squared Error regression elevation:', np.sqrt(metrics.mean_squared_error(y_test_z, y_pred_z)))


#----------------------- test for Q=600 

data600= pd.read_csv("Result_Q600_mesh1m_f.csv")
X_h600_in = data600[:][['X','Y','Q']]
y_h600_in = data600[:]['Depth']
X_wd600_in=data600[:][['X','Y','Q']]
y_wd600_in = data600[:]['IBC']
X_z600_in = data600[:][['X','Y']]
y_z600_in = data600[:]['Elevation']


X_train_h600, X_test_h600, y_train_h600, y_test_h600 = train_test_split(X_h600_in, y_h600_in, test_size=100, random_state=0)
X_train_wd600, X_test_wd600, y_train_wd600, y_test_wd600 = train_test_split(X_wd600_in, y_wd600_in, test_size=100, random_state=5)
X_train_z600, X_test_z600, y_train_z600, y_test_z600 = train_test_split(X_z600_in, y_z600_in, test_size=100, random_state=2)


X_h_sc600 = sc.transform(X_test_h600)

X_wd_sc600 = sc.transform(X_test_wd600)

X_z_sc600 = sc.transform(X_test_z600)


y_pred_h600 = regressor_h.predict(X_h_sc600)
#Evaluating h regression error
print('Mean Absolute Error regression depth for Q=600 :', metrics.mean_absolute_error(y_h600_in, y_pred_h600))
print('Mean Squared Error regression depth for Q=600 :', metrics.mean_squared_error(y_h600_in, y_pred_h600))
print('Root Mean Squared Error regression depth for Q=600 :', np.sqrt(metrics.mean_squared_error(y_h600_in, y_pred_h600)))
print('\n')
print('#-------------------Random forest for wet dry nodes for Q=600')

y_pred_wd600 = clf.predict(X_wd_sc600)
print('Mean Absolute Error classify wet dry:', metrics.mean_absolute_error(y_wd600_in, y_pred_wd600))
print('Mean Squared Error classify wet dry:', metrics.mean_squared_error(y_wd600_in, y_pred_wd600))
print('Root Mean Squared Error classify wet dry:', np.sqrt(metrics.mean_squared_error(y_wd600_in, y_pred_wd600)))
print('\n')

print('-------------------Random forest elevation prediction for Q=600\n')


y_pred_z600 = regressor_z.predict(X_z_sc600)
#Evaluating h regression error
print('Mean Absolute Error regression elevation:', metrics.mean_absolute_error(y_z600_in, y_pred_z600))
print('Mean Squared Error regression elevation:', metrics.mean_squared_error(y_z600_in, y_pred_z600))
print('Root Mean Squared Error regression elevation:', np.sqrt(metrics.mean_squared_error(y_z600_in, y_pred_z600)))







