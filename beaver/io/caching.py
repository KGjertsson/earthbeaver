from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from joblib import Parallel, delayed


from ..data_processing import feature_extraction

STATISTICAL_FEATURE_FILE = Path('statistical_feature_cache')


class FeatureReader:
    def __init__(self):
        self.df = pd.DataFrame()
        self.x_train = pd.DataFrame()
        self.y_train = pd.DataFrame()

    def generate(self, chunk):
        self.df = self.df.append(chunk)
        if len(self.df) >= 150000:
            df = self.df[-150000:]
            ch = feature_extraction.gen_statistical_features1(
                df['acoustic_data'])

            self.x_train = self.x_train.append(ch, ignore_index=True)
            self.y_train = self.y_train.append(
                pd.Series(df['time_to_failure'].values[-1]),
                ignore_index=True)
        return self.x_train, self.y_train


def load_statistical_data_parallel(root_data_dir):
    gen = FeatureReader()
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
        Parallel(
            n_jobs=8,
            backend='threading'
        )(delayed(gen.generate)(chunk) for chunk in tqdm(train))

        gen.x_train.to_csv(
            str(STATISTICAL_FEATURE_FILE) + '~x_train.csv', index=False)
        gen.y_train.to_csv(
            str(STATISTICAL_FEATURE_FILE) + '~y_train.csv', index=False)
        x_train = gen.x_train
        y_train = gen.y_train

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    return x_train, y_train, sc


def load_statistical_data(root_data_dir, feature_function_name):
    cache_file_base = feature_function_name + '_cache'

    if Path(cache_file_base + '~x_train.csv').exists():
        x_train = pd.read_csv(cache_file_base + '~x_train.csv')
        y_train = pd.read_csv(cache_file_base + '~y_train.csv')
    else:
        feature_function = getattr(feature_extraction, feature_function_name)
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
                ch = feature_function(df['acoustic_data'])
                x_train = x_train.append(ch, ignore_index=True)
                y_train = y_train.append(
                    pd.Series(df['time_to_failure'].values[-1]),
                    ignore_index=True)

        x_train.to_csv(cache_file_base + '~x_train.csv', index=False)
        y_train.to_csv(cache_file_base + '~y_train.csv', index=False)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    return x_train, y_train, sc
