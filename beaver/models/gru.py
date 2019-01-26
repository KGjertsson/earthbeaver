from time import time

from keras.layers import Dense, CuDNNGRU
from keras.optimizers import adam

from keras.models import Sequential


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
