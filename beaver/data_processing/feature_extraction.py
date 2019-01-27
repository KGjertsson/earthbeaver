import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.signal import hilbert
from scipy.signal import convolve
from scipy.signal import hann
from scipy import stats
from obspy.signal.trigger import recursive_sta_lta

import pandas as pd


def extract_features(z):
    """
    Helper function for the data generator. Basically we want to extract mean,
    standard deviation, min and max per time step.
    """
    return np.c_[z.mean(axis=1),
                 np.median(np.abs(z), axis=1),
                 z.std(axis=1),
                 z.max(axis=1),
                 z.min(axis=1)]


# For a given ending position "last_index", we split the last 150'000
# values of "x" into 150 pieces of length 1000 each.
# From each piece, 16 features are extracted. This results in a feature
# matrix of dimension (150 time steps x 16 features).
def create_x(x, last_index=None, n_steps=150, step_length=1000):
    if not last_index:
        last_index = len(x)
    assert last_index - n_steps * step_length >= 0

    # Reshaping and approximate standardization with mean 5 and std 3.
    feature_matrix = recursive_sta_lta(x[(last_index - n_steps * step_length):last_index], 20000, 150000)
    feature_matrix = feature_matrix.reshape(n_steps, -1)

    # Extracts features of sequences of full length 1000, of the last 100
    # values and finally also of the last 10 observations.
    return np.c_[extract_features(feature_matrix),
                 extract_features(feature_matrix[:, -step_length // 10:]),
                 extract_features(feature_matrix[:, -step_length // 100:]),
                 feature_matrix[:, -1:]]


def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta


def extract_multiple_features(train_frame, segment, rows):
    seg = train_frame.iloc[segment * rows:segment * rows + rows]
    x = pd.Series(seg['acoustic_data'].values)

    features = list()

    y = seg['time_to_failure'].values[-1]
    # seg.loc['time_to_failure'] = y

    features.append(x.mean())
    features.append(x.std())
    features.append(x.max())
    features.append(x.min())

    features.append(np.mean(np.diff(x)))
    features.append(np.mean(np.nonzero((np.diff(x) / x[:-1]))[0]))
    features.append(np.abs(x).max())
    features.append(np.abs(x).min())

    features.append(x[:50000].std())
    features.append(x[-50000:].std())
    features.append(x[:10000].std())
    features.append(x[-10000:].std())

    features.append(x[:50000].mean())
    features.append(x[-50000:].mean())
    features.append(x[:10000].mean())
    features.append(x[-10000:].mean())

    features.append(x[:50000].min())
    features.append(x[-50000:].min())
    features.append(x[:10000].min())
    features.append(x[-10000:].min())

    features.append(x[:50000].max())
    features.append(x[-50000:].max())
    features.append(x[:10000].max())
    features.append(x[-10000:].max())

    features.append(x.max() / np.abs(x.min()))
    features.append(x.max() - np.abs(x.min()))
    features.append(len(x[np.abs(x) > 500]))
    features.append(x.sum())

    features.append(np.mean(np.nonzero((np.diff(x[:50000]) / x[:50000][:-1]))[0]))
    features.append(np.mean(np.nonzero((np.diff(x[-50000:]) / x[-50000:][:-1]))[0]))
    features.append(np.mean(np.nonzero((np.diff(x[:10000]) / x[:10000][:-1]))[0]))
    features.append(np.mean(np.nonzero((np.diff(x[-10000:]) / x[-10000:][:-1]))[0]))

    features.append(np.quantile(x, 0.95))
    features.append(np.quantile(x, 0.99))
    features.append(np.quantile(x, 0.99))
    features.append(np.quantile(x, 0.01))

    features.append(np.quantile(np.abs(x), 0.95))
    features.append(np.quantile(np.abs(x), 0.99))
    features.append(np.quantile(np.abs(x), 0.05))
    features.append(np.quantile(np.abs(x), 0.01))

    features.append(add_trend_feature(x))
    features.append(add_trend_feature(x, abs_values=True))
    features.append(np.abs(x).mean())
    features.append(np.abs(x).std())

    features.append(x.mad())
    features.append(x.kurtosis())
    features.append(x.skew())
    features.append(x.median())

    features.append(np.abs(hilbert(x)).mean())
    features.append((convolve(x, hann(150), mode='same') / sum(hann(150))).mean())

    features.append(classic_sta_lta(x, 500, 10000).mean())
    features.append(classic_sta_lta(x, 5000, 100000).mean())
    features.append(classic_sta_lta(x, 3333, 6666).mean())
    features.append(classic_sta_lta(x, 10000, 25000).mean())
    features.append(classic_sta_lta(x, 50, 1000).mean())
    features.append(classic_sta_lta(x, 100, 5000).mean())
    features.append(classic_sta_lta(x, 333, 666).mean())
    features.append(classic_sta_lta(x, 4000, 10000).mean())

    moving_average_mean_700 = x.rolling(window=700).mean().mean(skipna=True)
    features.append(moving_average_mean_700)
    moving_average_mean_1500 = x.rolling(window=1500).mean().mean(skipna=True)
    features.append(moving_average_mean_1500)
    moving_average_mean_3000 = x.rolling(window=3000).mean().mean(skipna=True)
    features.append(moving_average_mean_3000)
    moving_average_mean_3000 = x.rolling(window=6000).mean().mean(skipna=True)
    features.append(moving_average_mean_3000)

    ewma = pd.Series.ewm
    features.append((ewma(x, span=300).mean()).mean(skipna=True))
    features.append(ewma(x, span=3000).mean().mean(skipna=True))
    features.append(ewma(x, span=6000).mean().mean(skipna=True))

    no_of_std = 2
    ma_700ma_std_mean = x.rolling(window=700).std().mean()
    features.append(ma_700ma_std_mean)
    features.append((moving_average_mean_700 + no_of_std * ma_700ma_std_mean).mean())
    features.append((moving_average_mean_700 - no_of_std * ma_700ma_std_mean).mean())

    ma_400ma_std_mean = x.rolling(window=400).std().mean()
    features.append(ma_400ma_std_mean)
    features.append((moving_average_mean_700 + no_of_std * ma_400ma_std_mean).mean())
    features.append((moving_average_mean_700 - no_of_std * ma_400ma_std_mean).mean())
    features.append(x.rolling(window=1000).std().mean())

    features.append(np.subtract(*np.percentile(x, [75, 25])))
    features.append(np.quantile(x, 0.999))
    features.append(np.quantile(x, 0.001))
    features.append(stats.trim_mean(x, 0.1))

    for windows in [10, 100, 1000]:
        x_roll_std = x.rolling(windows).std().dropna().values
        x_roll_mean = x.rolling(windows).mean().dropna().values

        features.append(x_roll_std.mean())
        features.append(x_roll_std.std())
        features.append(x_roll_std.max())
        features.append(x_roll_std.min())
        features.append(np.quantile(x_roll_std, 0.01))
        features.append(np.quantile(x_roll_std, 0.05))
        features.append(np.quantile(x_roll_std, 0.95))
        features.append(np.quantile(x_roll_std, 0.99))
        features.append(np.mean(np.diff(x_roll_std)))
        features.append(np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0]))
        features.append(np.abs(x_roll_std).max())

        features.append(x_roll_mean.mean())
        features.append(x_roll_mean.std())
        features.append(x_roll_mean.max())
        features.append(x_roll_mean.min())
        features.append(np.quantile(x_roll_mean, 0.01))
        features.append(np.quantile(x_roll_mean, 0.05))
        features.append(np.quantile(x_roll_mean, 0.95))
        features.append(np.quantile(x_roll_mean, 0.99))
        features.append(np.mean(np.diff(x_roll_mean)))
        features.append(np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0]))
        features.append(np.abs(x_roll_mean).max())

    return np.stack(features), y


