from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..data_processing import feature_extraction

STATISTICAL_FEATURE_FILE = Path('statistical_feature_cache')


def load_statistical_data(root_data_dir, val_split=None):
    if Path(str(STATISTICAL_FEATURE_FILE) + '~x_train.csv').exists():
        x_train = pd.read_csv(str(STATISTICAL_FEATURE_FILE) + '~x_train.csv')
        y_train = pd.read_csv(str(STATISTICAL_FEATURE_FILE) + '~y_train.csv')
    else:
        train = pd.read_csv(
            root_data_dir / 'lanl/train.csv',
            iterator=True,
            chunksize=30_000,
            dtype={'acoustic_data': np.int16,
                   'time_to_failure': np.float64})

        x_train = pd.DataFrame()
        y_train = pd.DataFrame()

        df = pd.DataFrame()
        for chunk in tqdm(train):
            df = df.append(chunk)
            if len(df) >= 150000:
                df = df[-150000:]
                ch = feature_extraction.gen_statistical_features(
                    df['acoustic_data'])
                x_train = x_train.append(ch, ignore_index=True)
                y_train = y_train.append(
                    pd.Series(df['time_to_failure'].values[-1]),
                    ignore_index=True)

        x_train.to_csv(
            str(STATISTICAL_FEATURE_FILE) + '~x_train.csv', index=False)
        y_train.to_csv(
            str(STATISTICAL_FEATURE_FILE) + '~y_train.csv', index=False)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    x_tr = x_train[:val_split]
    y_tr = y_train[:val_split]

    if val_split:
        x_val = x_train[val_split:]
        y_val = y_train[val_split:]
    else:
        x_val = pd.DataFrame()
        y_val = pd.DataFrame()

    return x_tr, y_tr, x_val, y_val, sc
