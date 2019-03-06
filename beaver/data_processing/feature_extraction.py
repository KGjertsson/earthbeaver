import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression


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


def gen_statistical_features1(train_chunk):
    features = []

    x_1 = train_chunk[:75000]
    x_2 = train_chunk[75000:]

    for segment in (x_1, x_2):
        features.append(segment.min())
        features.append(np.quantile(segment, 0.001))

        for i in range(1, 100):
            features.append(np.quantile(segment, i / 100))

        features.append(np.quantile(segment, 0.999))
        features.append(segment.max())
        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())

        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        features.append(np.sqrt(np.mean(segment ** 2)))

    return pd.Series(features)


def gen_statistical_features2(train_chunk):
    features = []
    train_chunk = pd.Series(data=savgol_filter(train_chunk, 10001, 2))
    x_1 = train_chunk[:75000]
    x_2 = train_chunk[75000:]

    for segment in (x_1, x_2):
        features.append(segment.min())
        features.append(np.quantile(segment, 0.001))

        for i in range(1, 100):
            features.append(np.quantile(segment, i / 100))

        features.append(np.quantile(segment, 0.999))
        features.append(segment.max())

        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())

        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        features.append(np.sqrt(np.mean(segment ** 2)))

    return pd.Series(features)


def gen_statistical_features3(train_chunk):
    # savgol_filter
    features = []

    x_1 = train_chunk[:75000]
    x_2 = train_chunk[75000:]

    for segment in (x_1, x_2):
        features.append(segment.min())
        features.append(np.quantile(segment, 0.001))

        for i in range(1, 100):
            features.append(np.quantile(segment, i / 100))

        features.append(np.quantile(segment, 0.999))
        features.append(segment.max())

        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())

        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        features.append(np.sqrt(np.mean(segment ** 2)))

    train_chunk = pd.Series(data=savgol_filter(train_chunk, 10001, 2))
    x_1 = train_chunk[:75000]
    x_2 = train_chunk[75000:]

    for segment in (x_1, x_2):
        features.append(segment.min())
        features.append(np.quantile(segment, 0.001))

        for i in range(1, 100):
            features.append(np.quantile(segment, i / 100))

        features.append(np.quantile(segment, 0.999))
        features.append(segment.max())

        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())

        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        features.append(np.sqrt(np.mean(segment ** 2)))

    return pd.Series(features)


def gen_statistical_features4(train_chunk):
    # add_trend_feature
    features = []

    x_1 = train_chunk[:75000]
    x_2 = train_chunk[75000:]

    for segment in (x_1, x_2):
        features.append(segment.min())
        features.append(np.quantile(segment, 0.001))

        for i in range(1, 100):
            features.append(np.quantile(segment, i / 100))

        features.append(np.quantile(segment, 0.999))
        features.append(segment.max())

        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())

        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        features.append(np.sqrt(np.mean(segment ** 2)))

        features.append(add_trend_feature(segment.values))
        features.append(add_trend_feature(segment.values, abs_values=True))

    train_chunk = pd.Series(data=savgol_filter(train_chunk, 10001, 2))
    x_1 = train_chunk[:75000]
    x_2 = train_chunk[75000:]

    for segment in (x_1, x_2):
        features.append(segment.min())
        features.append(np.quantile(segment, 0.001))

        for i in range(1, 100):
            features.append(np.quantile(segment, i / 100))

        features.append(np.quantile(segment, 0.999))
        features.append(segment.max())

        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())

        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        features.append(np.sqrt(np.mean(segment ** 2)))

        features.append(add_trend_feature(segment.values))
        features.append(add_trend_feature(segment.values, abs_values=True))

    return pd.Series(features)


def gen_statistical_features5(train_chunk):
    # removed add_trend_feature, added classic_sta_lta
    features = []

    x_1 = train_chunk[:75000]
    x_2 = train_chunk[75000:]

    for segment in (x_1, x_2):
        features.append(segment.min())
        features.append(np.quantile(segment, 0.001))

        for i in range(1, 100):
            features.append(np.quantile(segment, i / 100))

        features.append(np.quantile(segment, 0.999))
        features.append(segment.max())

        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())

        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        features.append(np.sqrt(np.mean(segment ** 2)))

        features.append(classic_sta_lta(segment.values, 500, 10000).mean())
        features.append(classic_sta_lta(segment.values, 5000, 100000).mean())
        features.append(classic_sta_lta(segment.values, 3333, 6666).mean())
        features.append(classic_sta_lta(segment.values, 10000, 25000).mean())

    train_chunk = pd.Series(data=savgol_filter(train_chunk, 10001, 2))
    x_1 = train_chunk[:75000]
    x_2 = train_chunk[75000:]

    for segment in (x_1, x_2):
        features.append(segment.min())
        features.append(np.quantile(segment, 0.001))

        for i in range(1, 100):
            features.append(np.quantile(segment, i / 100))

        features.append(np.quantile(segment, 0.999))
        features.append(segment.max())

        features.append(segment.mean())
        features.append(segment.std())
        features.append(segment.mad())
        features.append(segment.skew())
        features.append(segment.kurtosis())

        features.append(np.abs(segment).mean())
        features.append(np.abs(segment).std())
        features.append(np.sqrt(np.mean(segment ** 2)))

        features.append(classic_sta_lta(segment.values, 500, 10000).mean())
        features.append(classic_sta_lta(segment.values, 5000, 100000).mean())
        features.append(classic_sta_lta(segment.values, 3333, 6666).mean())
        features.append(classic_sta_lta(segment.values, 10000, 25000).mean())

    return pd.Series(features)