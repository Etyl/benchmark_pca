import numpy as np

from benchopt.benchmark import get_running_benchmark
from benchmark_utils.data import BaseDataset
from benchmark_utils.stiefel import uniform

def generate_data(n, d, rank, decay_function, decay_alpha, random_seed):
    W = uniform(d, rank, random_seed)
    if decay_function == "linear":
        S = 1 / (1 + decay_alpha * np.arange(rank))
    elif decay_function == "sqrt":
        S = 1 / (1 + np.sqrt(decay_alpha * np.arange(rank)))
    elif decay_function == "exp":
        S = np.exp(-decay_alpha * np.linspace(0, 1, rank))
    else:
        raise ValueError(f"Unknown decay function {decay_function}")
    V = uniform(n, rank, random_seed + 1)
    X = W @ np.diag(S) @ V.T
    return X, W

class Dataset(BaseDataset):
    """Simulated low rank matrix with ground truth PCA (up to numerical precision)."""

    name = "simulated"

    parameters = {
        "n": [100,1000],
        "d": [100, 1000],
        "rank": [10],
        "decay_function": ["linear", "sqrt", "exp"],
        "decay_alpha": [1],
        "random_seed": [14],
    }

    requirements = ["numpy", "scipy"]

    def get_data(self):
        benchmark = get_running_benchmark()
        if benchmark is None:
            raise RuntimeError(
                "This dataset can only be instantiated with a running benchmark."
            )
        generate_function = benchmark.cache(generate_data)
        X, W = generate_function(
            n = self.n,
            d = self.d,
            rank = self.rank,
            decay_function = self.decay_function,
            decay_alpha = self.decay_alpha,
            random_seed = self.random_seed,
        )
        return dict(X=X.T, W=W)