# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:04:56 2019

@author: Yihan Chen
"""


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


print('loading file...')



files = [currentpath for currentpath in Path('C:\\earthbeaver\\data\\lanl\\cycles').glob('*')]
pd.set_option("display.precision", 15)

for file in files:
    df = pd.read_csv(file)
    plt.figure(figsize=(20,10))
    print('plotting'+str(file)[32:-4])
    df[df.columns[1]].plot(grid=True, label=df.columns[1])
    #df[df.columns[2]].plot(grid=True, label=df.columns[2])
    plt.savefig('C:\\'+str(file)[32:-4])
    
    plt.close('all')
    plt.legend()
    








