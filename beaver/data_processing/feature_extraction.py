import numpy as np


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
