from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np


class AverageNet:

    def __init__(self, n_networks, network_function, *network_args,
                 **network_kwargs):
        self.networks = [network_function(*network_args, **network_kwargs)
                         for _ in range(n_networks)]

    def fit(self, x_tr, y_tr, validation_split, epochs, batch_size, callbacks):
        for model in self.networks:
            model.fit(
                x_tr,
                y_tr,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
            )

    def predict(self, x_tr):
        return np.average(
            [model.predict(x_tr) for model in self.networks],
            axis=0
        )


def simple_ffnn(x_tr, dropout_factor=0.25):
    model = Sequential()

    model.add(Dense(128, kernel_initializer='normal', input_dim=x_tr.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(dropout_factor))

    model.add(Dense(64, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(dropout_factor))

    model.add(Dense(32, kernel_initializer='normal'))
    model.add(Activation('tanh'))

    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Activation('linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam',
                  metrics=['mean_absolute_error'])

    return model


def make_heavy_ffn(x_tr):
    model = Sequential()

    model.add(Dense(4096, kernel_initializer='normal',
                    input_dim=x_tr.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))

    model.add(Dense(4096, kernel_initializer='normal'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Dropout(0.25))

    model.add(Dense(32, kernel_initializer='normal'))
    model.add(Activation('tanh'))

    model.add(Dense(1, kernel_initializer='normal'))
    model.add(Activation('linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam',
                  metrics=['mean_absolute_error'])

    return model
