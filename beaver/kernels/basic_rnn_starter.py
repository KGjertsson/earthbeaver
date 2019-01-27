# source: https://www.kaggle.com/mayer79/rnn-starter
from keras.models import load_model
import numpy as np
import pandas as pd

from ..models.gru import make_model
from ..data_processing.feature_extraction import create_x
from ..train import ann
from ..io import io_utils


# We call "extract_features" three times, so the total number of features
# is 3 * 5 + 1 (last value) = 16
def run_kernel(input_dir, n_features=16, batch_size=32):
    train_model = True

    if train_model:
        train_data = io_utils.load_train_data_numpy(input_dir)
        train_gen, _ = ann.init_generators(train_data, n_features, batch_size)
        model = make_model(n_features)
        history = ann.perform_training(model, train_gen, None)
        # plot_training_history(history)
        io_utils.make_submission(input_dir, model)

        print('Min val loss: {} at epoch: {}'.format(
            np.min(history.history['val_loss']),
            np.argmin(history.history['val_loss'])))
    else:
        train_data = io_utils.load_train_data_numpy(input_dir)

        model = load_model('model.hdf5')
        train_data_length = len(train_data)

        columns = ['end_index', 'time_to_boom', 'ground_truth']
        train_predictions = pd.DataFrame(columns=columns)

        for pandas_index, i in enumerate(range(0, train_data_length, 150000)):
            end_index = i + 150000
            if i + 150000 > train_data_length:
                end_index = train_data_length - 1
                i = end_index - 150000

            train_features = create_x(train_data[i:end_index, 0])
            train_features = np.expand_dims(train_features, 0)
            prediction = model.predict(train_features)[0][0]
            train_predictions.loc[pandas_index] = \
                [i, prediction, train_data[end_index, 1]]

        train_predictions.to_csv('train_predictions.csv')
        print(train_predictions)

# TODO: add https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient as loss
