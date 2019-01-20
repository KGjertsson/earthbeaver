# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:04:56 2019

@author: Yihan Chen
"""


import pandas as pd
import numpy as np


print('loading file...')
filename="C:\\Users\\Yihan Chen\\Desktop\\TestData\\train.csv"

pd.set_option("display.precision", 15)

train = pd.read_csv(filename, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
print('done')

cycle=1

print('splitting data')


eqp=[-1,5656573,50085877,104677355,138772452,187641819,218652629,245829584,307838916,338276286,375377847,419368879,461811622,495800224,528777114,585568143,621985672]


for i in range(len(eqp)):
    if i==-1:
        continue
    
    print('Printing cycle',i)
    filepath = 'C:\\Users\\Yihan Chen\\Desktop\\TestData\\'
    filename = 'cycle'+str(i)+'.csv'
    fullfile = filepath + filename
    train[eqp[i-1]+1:eqp[i]].to_csv(fullfile)
    print('Done')    


