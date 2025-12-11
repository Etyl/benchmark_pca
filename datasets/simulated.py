import numpy as np
import os

from benchopt.benchmark import get_running_benchmark
from benchmark_utils.data import BaseDataset
from benchmark_utils.stiefel import uniform
from benchopt.config import get_data_path


def generate_data(
    X_path, n, d, rank, decay_function, decay_alpha=1, random_seed=0
):
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

    np.save(X_path, X)


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
    }

    requirements = ["numpy", "scipy"]

    def get_data(self):
        benchmark = get_running_benchmark()
        if benchmark is None:
            raise RuntimeError(
                "This dataset can only be instantiated "
                "with a running benchmark."
            )

        X_file_name = (
            f"X_{self.n}_{self.d}_{self.rank}_{self.decay_function}.npy"
        )
        X_dir = os.path.join(get_data_path(), "simulated")
        X_path = os.path.join(X_dir, X_file_name)

        if not os.path.exists(X_path):
            os.makedirs(X_dir, exist_ok=True)
            generate_data(
                X_path=X_path,
                n=self.n,
                d=self.d,
                rank=self.rank,
                decay_function=self.decay_function,
            )

        return dict(
            n=self.n,
            d=self.d,
            X_path=X_path,
        )
