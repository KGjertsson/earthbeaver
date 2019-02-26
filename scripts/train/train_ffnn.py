from pathlib import Path
from time import time

import pandas as pd
from keras import callbacks
from sklearn.metrics import mean_absolute_error

from beaver.data_processing import feature_extraction
from beaver.io import caching
from beaver.models import feed_forward_nets


def submit(root_data_dir, sc, model, feature_function_name, network_type,
           dropout_factor, epochs, n_networks=1):
    submission = pd.read_csv(
        root_data_dir / 'lanl/sample_submission.csv', index_col='seg_id')
    x_test = pd.DataFrame()

    cache_file = Path(feature_function_name + '_cache~x_test.csv')
    if cache_file.exists():
        x_test = pd.read_csv(cache_file)
    else:
        feature_function = getattr(feature_extraction, feature_function_name)
        for seg_id in submission.index:
            seg = pd.read_csv(
                root_data_dir / ('lanl/test/' + seg_id + '.csv'))
            ch = feature_function(seg['acoustic_data'])
            x_test = x_test.append(ch, ignore_index=True)
            x_test.to_csv(cache_file, index=False)

    x_test = sc.transform(x_test)
    prediction = model.predict(x_test).clip(0, 16.1)

    submission['time_to_failure'] = prediction

    submission_file = 'submission~' + \
                      network_type + \
                      '~epochs_' + \
                      str(epochs) + \
                      '~dropout_' + \
                      str(dropout_factor) + \
                      '~' + \
                      feature_function_name
    if n_networks > 1:
        submission_file = submission_file + '~n_networks=' + str(n_networks)
    submission_file = submission_file + '.csv'

    submission.to_csv(submission_file)


def main():
    root_data_dir = Path('../../data/')
    dropout_factor = 0.5
    feature_function = 'gen_statistical_features3'
    network_type = 'simple_ffnn'
    epochs = 3
    n_networks = 100

    x_tr, y_tr, sc = caching.load_statistical_data(
        root_data_dir=root_data_dir,
        feature_function_name=feature_function)

    # model_function = getattr(feed_forward_nets, network_type)
    # model = model_function(x_tr, dropout_factor)

    model = feed_forward_nets.AverageNet(
        n_networks,
        getattr(feed_forward_nets, network_type),
        x_tr,
        dropout_factor
    )

    model.fit(
        x_tr,
        y_tr.values,
        validation_split=0.,
        epochs=epochs,
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

    submit(
        root_data_dir,
        sc,
        model,
        feature_function,
        network_type,
        dropout_factor,
        epochs,
        n_networks)


if __name__ == '__main__':
    main()
