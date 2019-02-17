from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from keras import callbacks
from sklearn.metrics import mean_absolute_error

from beaver.data_processing import feature_extraction
from beaver.io import caching
from beaver.models import feed_forward_nets


def main():
    root_data_dir = Path('../../data/')

    x_tr, y_tr, sc = caching.load_statistical_data(root_data_dir)

    model = feed_forward_nets.make_simple_ffn(x_tr)
    # model = feed_forward_nets.make_heavy_ffn(x_tr)
    model.fit(
        x_tr,
        y_tr.values,
        validation_split=0.,
        epochs=1000,
        batch_size=32,
        callbacks=[
            callbacks.TensorBoard(
                log_dir='./Graph/' + str(time()), histogram_freq=0,
                write_graph=True, write_images=False)
        ]
    )
    cur_preds = model.predict(x_tr).clip(0, 16.1)
    train_score = mean_absolute_error(y_tr, cur_preds)

    print(train_score)

    submission = pd.read_csv(
        root_data_dir / 'lanl/sample_submission.csv', index_col='seg_id')
    x_test = pd.DataFrame()

    for seg_id in submission.index:
        seg = pd.read_csv(root_data_dir / ('lanl/test/' + seg_id + '.csv'))
        ch = feature_extraction.gen_statistical_features(seg['acoustic_data'])
        x_test = x_test.append(ch, ignore_index=True)

    x_test = sc.transform(x_test)
    prediction = model.predict(x_test).clip(0, 16.1)

    submission['time_to_failure'] = prediction
    submission.to_csv('submission.csv')


if __name__ == '__main__':
    main()
