from pathlib import Path

from beaver.kernels import basic_feature_benchmark
from beaver.kernels import basic_rnn_starter
from beaver.kernels import more_features_and_samples
from beaver.kernels import kernel_145

if __name__ == '__main__':
    input_dir = Path('../data/lanl')

    # basic_feature_benchmark.run_kernel(input_dir, verbose=True)
    # basic_rnn_starter.run_kernel(input_dir)
    # more_features_and_samples.run_kernel(input_dir)
    kernel_145.run_kernel()
