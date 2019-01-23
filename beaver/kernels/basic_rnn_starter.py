# source: https://www.kaggle.com/mayer79/rnn-starter
from time import time

from keras.models import Sequential, load_model
from keras.layers import Dense, CuDNNGRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
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
    feature_matrix = \
        x[(last_index - n_steps * step_length):last_index].reshape(n_steps, -1)

    # Extracts features of sequences of full length 1000, of the last 100
    # values and finally also of the last 10 observations.
    return np.c_[extract_features(feature_matrix),
                 extract_features(feature_matrix[:, -step_length // 10:]),
                 extract_features(feature_matrix[:, -step_length // 100:]),
                 feature_matrix[:, -1:]]


# The generator randomly selects "batch_size" ending positions of sub-time
# series. For each ending position,
# the "time_to_failure" serves as target, while the features are created
# by the function "create_x".
def generator(data, n_features, min_index=0, max_index=None, batch_size=16,
              n_steps=150, step_length=1000):
    if max_index is None:
        max_index = len(data) - 1

    while True:
        # Pick indices of ending positions
        rows = np.random.randint(min_index + n_steps * step_length,
                                 max_index, size=batch_size)

        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, n_features))
        targets = np.zeros(batch_size, )

        for j, row in enumerate(rows):
            samples[j] = create_x(data[:, 0], last_index=row,
                                  n_steps=n_steps, step_length=step_length)
            targets[j] = data[row, 1]
        yield samples, targets


# Visualize accuracies
def plot_training_history(history, what='loss'):
    x = history.history[what]
    val_x = history.history['val_' + what]
    epochs = np.asarray(history.epoch) + 1

    plt.plot(epochs, x, label="Training " + what)
    plt.plot(epochs, val_x, label="Validation " + what)
    plt.title("Training and validation " + what)
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()
    return None


def load_train_data(input_dir):
    print('Loading train data...')
    t_load = time()
    train_data = pd.read_csv(
        input_dir / "train.csv",
        dtype={"acoustic_data": np.float32, "time_to_failure": np.float32})
    train_data = train_data.values
    print('Done {} s'.format(time() - t_load))
    return train_data


def init_generators(train_data, n_features, batch_size):
    train_gen = generator(train_data, n_features, batch_size=batch_size)
    valid_gen = generator(train_data, n_features, batch_size=batch_size)
    return train_gen, valid_gen


def make_model(n_features):
    print('Creating model...')
    t_make_model = time()
    model = Sequential()
    model.add(CuDNNGRU(64,
                       # return_sequences=True,
                       input_shape=(None, n_features)))
    # model.add(CuDNNGRU(64))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # TODO: investigate linear activation function in final layer
    model.summary()

    model.compile(optimizer=adam(lr=5e-4), loss="mae")
    print('Done {} s'.format(time() - t_make_model))
    return model


def perform_training(model, train_gen, valid_gen):
    callbacks = [ModelCheckpoint("model.hdf5", monitor='val_loss',
                                 save_weights_only=False, period=3)]
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=1000,
                                  # n_train // batch_size,
                                  epochs=30,
                                  verbose=1,
                                  callbacks=callbacks,
                                  validation_data=valid_gen,
                                  # n_valid // batch_size)
                                  validation_steps=100)
    return history


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


# We call "extract_features" three times, so the total number of features
# is 3 * 5 + 1 (last value) = 16
def run_kernel(input_dir, n_features=16, batch_size=32):
    train_model = True

    if train_model:
        train_data = load_train_data(input_dir)
        train_gen, valid_gen = init_generators(train_data, n_features,
                                               batch_size)
        model = make_model(n_features)
        history = perform_training(model, train_gen, valid_gen)
        # plot_training_history(history)
        make_submission(input_dir, model)

        print('Min val loss: {} at epoch: {}'.format(
            np.min(history.history['val_loss']),
            np.argmin(history.history['val_loss'])))
    else:
        train_data = load_train_data(input_dir)

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
