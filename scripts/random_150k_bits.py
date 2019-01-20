# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 13:35:57 2019

@author: Yihan Chen
"""
import numpy as np
import pandas as pd
import random


input_dir = "C:\\Users\\Yihan Chen\\Desktop\\TestData\\"


train = pd.read_csv(
    input_dir+'train.csv',
    dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


length=len(train)

while 1==1:
    i=random.randint(0,length-150000)
    train[i:i+150000].to_csv('C:\\Users\\Yihan Chen\\Desktop\\TestData\\random'+str(i)+'.csv')
    print('printed at',i)










