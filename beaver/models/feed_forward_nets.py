from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization


def make_simple_ffn(x_tr):
    NN_model = Sequential()

    NN_model.add(
        Dense(128, kernel_initializer='normal', input_dim=x_tr.shape[1]))
    NN_model.add(BatchNormalization())
    NN_model.add(Activation('tanh'))
    NN_model.add(Dropout(0.25))

    NN_model.add(Dense(64, kernel_initializer='normal'))
    NN_model.add(BatchNormalization())
    NN_model.add(Activation('tanh'))
    NN_model.add(Dropout(0.25))

    NN_model.add(Dense(32, kernel_initializer='normal'))
    NN_model.add(Activation('tanh'))

    NN_model.add(Dense(1, kernel_initializer='normal'))
    NN_model.add(Activation('linear'))

    NN_model.compile(loss='mean_absolute_error', optimizer='adam',
                     metrics=['mean_absolute_error'])

    return NN_model
