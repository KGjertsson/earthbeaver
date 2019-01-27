import numpy as np

from ..io import io_utils
from ..train import ann
from ..models import gru


def train_rnn_with_multi_features(input_dir):
    train_frame = io_utils.load_train_data_pandas(input_dir)
    rows = 150_000
    segments = int(np.floor(train_frame.shape[0] / rows))

    train_generator = \
        ann.multi_feature_generator(train_frame, segments, rows, 32)

    rnn_model = gru.make_model(n_features=142)

    history = ann.perform_training(rnn_model, train_generator, None)
    return history
