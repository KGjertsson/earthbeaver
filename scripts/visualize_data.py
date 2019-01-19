from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

if __name__ == '__main__':
    input_dir = Path('../data/lanl/')

    train = pd.read_csv(
        input_dir / 'train.csv',
        dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

    train[train.columns[1]].plot()
    plt.savefig('data')
    plt.show()
