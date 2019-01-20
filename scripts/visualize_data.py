from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

if __name__ == '__main__':
    input_dir = Path('../data/lanl/')

    train = pd.read_csv(
        input_dir / 'train.csv',
        dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

    plt.figure(figsize=(50,30))
    train[train.columns[1]].plot(grid=True,label=train.columns[1])
    train[train.columns[0]].plot(grid=True,label=train.columns[0])
    
    plt.legend()
    plt.savefig('data_raw')
    plt.show()
    
    
    #preparations for LTA/STA processsed data
    plt.figure(figsize=(50,30))
    plt.plot(train[train.columns[0]],train[train.columns[1]],'.',color='b')
    linje=np.linspace(0,16,17)
    plt.plot(linje,linje,color='r')
    plt.savefig('data_scatter')
    
