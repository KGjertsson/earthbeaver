# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:04:56 2019

@author: Yihan Chen
"""


import pandas as pd
import matplotlib.pyplot as plt

print('loading file...')
filename="C:\\Users\\Yihan Chen\\Desktop\\TestData\\cycle9.csv"

pd.set_option("display.precision", 15)

train = pd.read_csv(filename)
print('done')


plt.figure(figsize=(20,10))
train[train.columns[1]].plot()



