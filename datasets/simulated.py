import numpy as np
import tempfile
import os

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

    tmp_dir = tempfile.mkdtemp(prefix="benchopt_simulated_")
    X_path = os.path.join(tmp_dir, "X.npy")
    np.save(X_path, X)

    return X_path, W


class Dataset(BaseDataset):
    """
    Simulated low rank matrix with ground truth PCA
    (up to numerical precision).
    """

    name = "simulated"

    parameters = {
        "n": [10000],
        "d": [30000],
        "rank": [1000],
        "decay_function": ["sqrt"],
        "decay_alpha": [1],
        "random_seed": [14],
    }

    requirements = ["numpy", "scipy"]

    def get_data(self):
        benchmark = get_running_benchmark()
        if benchmark is None:
            raise RuntimeError(
                "This dataset can only be instantiated "
                "with a running benchmark."
            )
        X_path, W = generate_data(
            n=self.n,
            d=self.d,
            rank=self.rank,
            decay_function=self.decay_function,
            decay_alpha=self.decay_alpha,
            random_seed=self.random_seed,
        )
        return dict(
            n=self.n,
            d=self.d,
            X_path=X_path,
            W=W
        )
