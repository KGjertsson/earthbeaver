import pandas as pd

import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('train_predictions.csv')

    plt.plot(data['time_to_boom'])
    plt.plot(data['ground_truth'])
    plt.grid()
    plt.show()
