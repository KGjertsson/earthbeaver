from pathlib import Path

from beaver.train import multi_feature_rnn

if __name__ == '__main__':
    input_dir = Path('../data/lanl')
    history = multi_feature_rnn.train_rnn_with_multi_features(input_dir)
