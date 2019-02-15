import os
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from scipy import stats
from scipy.signal import hann
from scipy.signal import hilbert
from scipy.signal import convolve
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from beaver.data_processing import features_145, gpi


def main():
    data_root_dir = '../../data/lanl/'
    train_df = pd.read_csv(
        data_root_dir+'train.csv',
        dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32}
    )
    rows = 150000
    segments = int(np.floor(train_df.shape[0] / rows))

    if Path('cached_features_X.csv').exists():
        train_X = pd.read_csv('cached_features_X.csv')
        train_y = pd.read_csv('cached_labels_y.csv')
    else:
        train_X = pd.DataFrame(index=range(segments), dtype=np.float64)
        train_y = pd.DataFrame(
            index=range(segments),
                               dtype=np.float64,
                               columns=['time_to_failure'])

        total_mean = train_df['acoustic_data'].mean()
        total_std = train_df['acoustic_data'].std()
        total_max = train_df['acoustic_data'].max()
        total_min = train_df['acoustic_data'].min()
        total_sum = train_df['acoustic_data'].sum()
        total_abs_sum = np.abs(train_df['acoustic_data']).sum()

        # iterate over all segments
        features_145.add_features_to_dataframe(train_y, train_X, train_df, segments, rows)

    scaler = StandardScaler()
    scaler.fit(train_X)
    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    submission = pd.read_csv('../../data/lanl/sample_submission.csv', index_col='seg_id')
    test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)

    for _, seg_id in enumerate(tqdm(test_X.index)):
        seg = pd.read_csv('../../data/lanl/test/' + seg_id + '.csv')

        xc = pd.Series(seg['acoustic_data'].values)
        zc = np.fft.fft(xc)
        test_X.loc[seg_id, 'mean'] = xc.mean()
        test_X.loc[seg_id, 'std'] = xc.std()
        test_X.loc[seg_id, 'max'] = xc.max()
        test_X.loc[seg_id, 'min'] = xc.min()

        #FFT transform values
        realFFT = np.real(zc)
        imagFFT = np.imag(zc)
        test_X.loc[seg_id, 'A0'] = abs(zc[0])
        test_X.loc[seg_id, 'Rmean'] = realFFT.mean()
        test_X.loc[seg_id, 'Rstd'] = realFFT.std()
        test_X.loc[seg_id, 'Rmax'] = realFFT.max()
        test_X.loc[seg_id, 'Rmin'] = realFFT.min()
        test_X.loc[seg_id, 'Imean'] = imagFFT.mean()
        test_X.loc[seg_id, 'Istd'] = imagFFT.std()
        test_X.loc[seg_id, 'Imax'] = imagFFT.max()
        test_X.loc[seg_id, 'Imin'] = imagFFT.min()

        test_X.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(xc))
        test_X.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(xc) / xc[:-1]))[0])
        test_X.loc[seg_id, 'abs_max'] = np.abs(xc).max()
        test_X.loc[seg_id, 'abs_min'] = np.abs(xc).min()

        test_X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()
        test_X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()
        test_X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()
        test_X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()

        test_X.loc[seg_id, 'avg_first_50000'] = xc[:50000].mean()
        test_X.loc[seg_id, 'avg_last_50000'] = xc[-50000:].mean()
        test_X.loc[seg_id, 'avg_first_10000'] = xc[:10000].mean()
        test_X.loc[seg_id, 'avg_last_10000'] = xc[-10000:].mean()

        test_X.loc[seg_id, 'min_first_50000'] = xc[:50000].min()
        test_X.loc[seg_id, 'min_last_50000'] = xc[-50000:].min()
        test_X.loc[seg_id, 'min_first_10000'] = xc[:10000].min()
        test_X.loc[seg_id, 'min_last_10000'] = xc[-10000:].min()

        test_X.loc[seg_id, 'max_first_50000'] = xc[:50000].max()
        test_X.loc[seg_id, 'max_last_50000'] = xc[-50000:].max()
        test_X.loc[seg_id, 'max_first_10000'] = xc[:10000].max()
        test_X.loc[seg_id, 'max_last_10000'] = xc[-10000:].max()

        test_X.loc[seg_id, 'max_to_min'] = xc.max() / np.abs(xc.min())
        test_X.loc[seg_id, 'max_to_min_diff'] = xc.max() - np.abs(xc.min())
        test_X.loc[seg_id, 'count_big'] = len(xc[np.abs(xc) > 500])
        test_X.loc[seg_id, 'sum'] = xc.sum()

        test_X.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(xc[:50000]) / xc[:50000][:-1]))[0])
        test_X.loc[seg_id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(xc[-50000:]) / xc[-50000:][:-1]))[0])
        test_X.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(xc[:10000]) / xc[:10000][:-1]))[0])
        test_X.loc[seg_id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(xc[-10000:]) / xc[-10000:][:-1]))[0])

        test_X.loc[seg_id, 'q95'] = np.quantile(xc,0.95)
        test_X.loc[seg_id, 'q99'] = np.quantile(xc,0.99)
        test_X.loc[seg_id, 'q05'] = np.quantile(xc,0.05)
        test_X.loc[seg_id, 'q01'] = np.quantile(xc,0.01)

        test_X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)
        test_X.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)
        test_X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)
        test_X.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)

        test_X.loc[seg_id, 'trend'] = features_145.add_trend_feature(xc)
        test_X.loc[seg_id, 'abs_trend'] = features_145.add_trend_feature(xc, abs_values=True)
        test_X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()
        test_X.loc[seg_id, 'abs_std'] = np.abs(xc).std()

        test_X.loc[seg_id, 'mad'] = xc.mad()
        test_X.loc[seg_id, 'kurt'] = xc.kurtosis()
        test_X.loc[seg_id, 'skew'] = xc.skew()
        test_X.loc[seg_id, 'med'] = xc.median()

        test_X.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()
        test_X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()
        test_X.loc[seg_id, 'classic_sta_lta1_mean'] = features_145.classic_sta_lta(xc, 500, 10000).mean()
        test_X.loc[seg_id, 'classic_sta_lta2_mean'] = features_145.classic_sta_lta(xc, 5000, 100000).mean()
        test_X.loc[seg_id, 'classic_sta_lta3_mean'] = features_145.classic_sta_lta(xc, 3333, 6666).mean()
        test_X.loc[seg_id, 'classic_sta_lta4_mean'] = features_145.classic_sta_lta(xc, 10000, 25000).mean()
        test_X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)
        test_X.loc[seg_id, 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)
        test_X.loc[seg_id, 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)
        test_X.loc[seg_id, 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)
        ewma = pd.Series.ewm
        test_X.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(xc, span=300).mean()).mean(skipna=True)
        test_X.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)
        test_X.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)
        no_of_std = 2
        test_X.loc[seg_id, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()
        test_X.loc[seg_id,'MA_700MA_BB_high_mean'] = (test_X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * test_X.loc[seg_id, 'MA_700MA_std_mean']).mean()
        test_X.loc[seg_id,'MA_700MA_BB_low_mean'] = (test_X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * test_X.loc[seg_id, 'MA_700MA_std_mean']).mean()
        test_X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()
        test_X.loc[seg_id,'MA_400MA_BB_high_mean'] = (test_X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * test_X.loc[seg_id, 'MA_400MA_std_mean']).mean()
        test_X.loc[seg_id,'MA_400MA_BB_low_mean'] = (test_X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * test_X.loc[seg_id, 'MA_400MA_std_mean']).mean()
        test_X.loc[seg_id, 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()

        test_X.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))
        test_X.loc[seg_id, 'q999'] = np.quantile(xc,0.999)
        test_X.loc[seg_id, 'q001'] = np.quantile(xc,0.001)
        test_X.loc[seg_id, 'ave10'] = stats.trim_mean(xc, 0.1)

        for windows in [10, 100, 1000]:
            x_roll_std = xc.rolling(windows).std().dropna().values
            x_roll_mean = xc.rolling(windows).mean().dropna().values

            test_X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
            test_X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
            test_X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
            test_X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
            test_X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
            test_X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
            test_X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
            test_X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
            test_X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
            test_X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            test_X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

            test_X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
            test_X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
            test_X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
            test_X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
            test_X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
            test_X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
            test_X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
            test_X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
            test_X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
            test_X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            test_X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)
    mean_absolute_error(train_y, gpi.GPI(scaled_train_X))
    predictions = gpi.GPI(scaled_test_X).values

    submission.time_to_failure = predictions
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
