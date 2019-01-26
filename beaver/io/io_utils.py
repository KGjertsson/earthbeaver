from time import time

import numpy as np
import pandas as pd

from ..data_processing.feature_extraction import create_x


def load_train_data(input_dir):
    print('Loading train data...')
    t_load = time()
    train_data = pd.read_csv(
        input_dir / "train.csv",
        dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
    train_data = train_data.values
    print('Done {} s'.format(time() - t_load))
    return train_data


def make_submission(input_dir, model):
    # Load submission file
    submission = pd.read_csv(
        input_dir / 'sample_submission.csv',
        index_col='seg_id', dtype={"time_to_failure": np.float32})

    # Load each test data, create the feature matrix, get numeric prediction
    for i, seg_id in enumerate(submission.index):
        seg = pd.read_csv(input_dir / ('test/' + seg_id + '.csv'))
        x = seg['acoustic_data'].values
        submission.time_to_failure[i] = model.predict(
            np.expand_dims(create_x(x), 0))

    submission.head()

    # Save
    submission.to_csv('submission.csv')
