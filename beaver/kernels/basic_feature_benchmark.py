import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR
from sklearn.metrics import mean_absolute_error
from pathlib import Path
from tqdm import tqdm


def run_kernel(input_dir, verbose=False):
    if verbose:
        print(os.listdir(input_dir))

    train = pd.read_csv(
        input_dir / 'train.csv',
        dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

    if verbose:
        print(train.head())

        pd.options.display.precision = 15

        print(train.head())

    # Create a training file with simple derived features

    rows = 150_000
    segments = int(np.floor(train.shape[0] / rows))

    X_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                           columns=['ave', 'std', 'max', 'min'])
    y_train = pd.DataFrame(index=range(segments), dtype=np.float64,
                           columns=['time_to_failure'])

    for segment in tqdm(range(segments)):
        seg = train.iloc[segment * rows:segment * rows + rows]
        x = seg['acoustic_data'].values
        y = seg['time_to_failure'].values[-1]

        y_train.loc[segment, 'time_to_failure'] = y

        X_train.loc[segment, 'ave'] = x.mean()
        X_train.loc[segment, 'std'] = x.std()
        X_train.loc[segment, 'max'] = x.max()
        X_train.loc[segment, 'min'] = x.min()

    if verbose:
        print(X_train.head())

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    svm = NuSVR()
    svm.fit(X_train_scaled, y_train.values.flatten())
    y_pred = svm.predict(X_train_scaled)

    if verbose:
        plt.figure(figsize=(6, 6))
        plt.scatter(y_train.values.flatten(), y_pred)
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.xlabel('actual', fontsize=12)
        plt.ylabel('predicted', fontsize=12)
        plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])
        plt.show()

    score = mean_absolute_error(y_train.values.flatten(), y_pred)

    if verbose:
        print(f'Score: {score:0.3f}')

    submission = pd.read_csv(
        input_dir / 'sample_submission.csv', index_col='seg_id')

    X_test = pd.DataFrame(columns=X_train.columns,
                          dtype=np.float64, index=submission.index)

    for seg_id in X_test.index:
        seg = pd.read_csv(input_dir / ('test/' + seg_id + '.csv'))

        x = seg['acoustic_data'].values

        X_test.loc[seg_id, 'ave'] = x.mean()
        X_test.loc[seg_id, 'std'] = x.std()
        X_test.loc[seg_id, 'max'] = x.max()
        X_test.loc[seg_id, 'min'] = x.min()

    X_test_scaled = scaler.transform(X_test)
    submission['time_to_failure'] = svm.predict(X_test_scaled)
    submission.to_csv('submission.csv')
