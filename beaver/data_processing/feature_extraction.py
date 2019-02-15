import numpy as np
import pandas as pd


def gen_statistical_features(train_chunk):
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
