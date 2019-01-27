from keras.callbacks import ModelCheckpoint
import numpy as np

from obspy.signal.trigger import recursive_sta_lta

from ..data_processing.feature_extraction import create_x, \
    extract_multiple_features


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

        # data[:, 0] = recursive_sta_lta(data[:, 0], 20000, 150000)

        for j, row in enumerate(rows):
            samples[j] = create_x(data[:, 0], last_index=row,
                                  n_steps=n_steps, step_length=step_length)
            targets[j] = data[row, 1]
        yield samples, targets


def multi_feature_generator(x_frame, rows, segments, batch_size):
    while True:
        x_batch = []
        y_batch = []

        batch_indices = list()
        for batch_index in range(batch_size):
            segment = np.random.randint(0, segments)
            while segment in batch_indices:
                segment = np.random.randint(0, segments)
            batch_indices.append(segment)

            x_features, y = extract_multiple_features(x_frame, segment, rows)

            x_batch.append(x_features)
            y_batch.append(y)

        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)

        yield x_batch, y_batch


def init_generators(train_data, n_features, batch_size):
    train_gen = generator(train_data, n_features, batch_size=batch_size)
    valid_gen = generator(train_data, n_features, batch_size=batch_size)
    return train_gen, valid_gen


def perform_training(model, train_gen, valid_gen):
    if valid_gen:
        validation_steps = 100
    else:
        validation_steps = None

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
                                  validation_steps=validation_steps)
    return history
