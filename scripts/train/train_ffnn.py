from pathlib import Path

import numpy as np
import pandas as pd
from keras import callbacks
from sklearn.metrics import mean_absolute_error

from beaver.data_processing import feature_extraction
from beaver.io import caching
from beaver.models import feed_forward_nets


def main():
    root_data_dir = Path('../../data/')

    x_tr, y_tr, _, _, sc = caching.load_statistical_data(root_data_dir)

    simple_ffnn_model = feed_forward_nets.make_simple_ffn(x_tr)
    simple_ffnn_model.fit(
        x_tr,
        y_tr.values,
        epochs=100,
        batch_size=32,
        callbacks=[
            callbacks.TensorBoard(
                log_dir='./Graph', histogram_freq=0,
                write_graph=True, write_images=False)
        ]
    )
    cur_preds = simple_ffnn_model.predict(x_tr).clip(0, 16.1)
    train_score = mean_absolute_error(y_tr, cur_preds)

    print(train_score)

    # submission = pd.read_csv(
    #     root_data_dir / 'lanl/sample_submission.csv', index_col='seg_id')
    # x_test = pd.DataFrame()
    #
    # for seg_id in submission.index:
    #     seg = pd.read_csv('../../data/lanl/test/' + seg_id + '.csv')
    #     ch = feature_extraction.gen_statistical_features(seg['acoustic_data'])
    #     x_test = x_test.append(ch, ignore_index=True)
    #
    # x_test = sc.transform(x_test)
    # preds = []
    # for _ in range(10):
    #     simple_ffnn_model = feed_forward_nets.make_simple_ffn(x_tr)
    #     simple_ffnn_model.fit(x_tr, y_tr, epochs=3, batch_size=32)
    #     cur_preds = simple_ffnn_model.predict(x_test).clip(0, 16.1)
    #     preds.append(cur_preds)
    #
    # preds = np.hstack([p.reshape(-1, 1) for p in preds])
    # preds = np.mean(preds, axis=1)
    #
    # submission['time_to_failure'] = preds
    # submission.to_csv('submission.csv')


if __name__ == '__main__':
    main()
