from pathlib import Path

from beaver.kernels import basic_feature_benchmark

if __name__ == '__main__':
    input_dir = Path('../data/lanl')
    basic_feature_benchmark.run_kernel(input_dir, verbose=True)