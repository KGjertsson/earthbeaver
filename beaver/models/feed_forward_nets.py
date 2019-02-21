from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization


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
