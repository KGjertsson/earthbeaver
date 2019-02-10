# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 13:04:56 2019

@author: Yihan Chen
"""


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from obspy.signal.trigger import recursive_sta_lta

print('loading file...')


files = [currentpath for currentpath in Path('C:\\earthbeaver\\data\\lanl\\cycles').glob('*')]
pd.set_option("display.precision", 15)

steps_spike_to_failure=[]
time_spike_to_failure=[]

for file in files:
    
    df = pd.read_csv(file)
    
    '''
    max_index=df['acoustic_data'].idxmax()
    print (str(file)[32:-4])
    steps_spike_to_failure.append(len(df['acoustic_data'])-max_index)
    time_spike_to_failure.append(df['time_to_failure'][max_index])
    '''
    

    plt.figure(figsize=(20,10))
    print('plotting'+str(file)[32:-4])
    #df[df.columns[1]].plot(grid=True, label=df.columns[1])
    #df[df.columns[2]].plot(grid=True, label=df.columns[2])
    x=recursive_sta_lta(df['acoustic_data'],20000,150000)
    plt.plot(x)
    plt.savefig('C:\\recursive'+str(file)[32:-4])
    
    plt.close('all')
    plt.legend()
    







