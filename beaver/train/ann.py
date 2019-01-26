from keras.callbacks import ModelCheckpoint
import numpy as np

from ..data_processing.feature_extraction import create_x


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


def init_generators(train_data, n_features, batch_size):
    train_gen = generator(train_data, n_features, batch_size=batch_size)
    valid_gen = generator(train_data, n_features, batch_size=batch_size)
    return train_gen, valid_gen


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
