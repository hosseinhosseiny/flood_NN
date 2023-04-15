# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 18:54:26 2019

@author: shossein
"""


import pandas as pd
import os
directory= os.getcwd()  # the current working directory          
from os import listdir
#
filepaths = [f for f in listdir(directory) if f.endswith('.csv')]
df = pd.concat(map(pd.read_csv, filepaths))           
pd.DataFrame(df).to_csv("test_.csv")


####################### checking the dimensions of result


import os
import pandas as pd
directory= os.getcwd()  # the current working directory   
df2= pd.read_csv('results.csv')
