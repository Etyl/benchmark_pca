import os
from benchopt import safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchmark_utils.data import DiskDataset
    from benchmark_utils.stiefel import uniform


class Dataset(DiskDataset):
    """Simulated low rank matrix with ground truth PCA (up to numerical precision)."""

    name = "simulated"

    parameters = {
        "n, d": [
            (100, 50),
            (500, 20),
        ],
        "rank": [10],
        "decay_function": ["linear", "sqrt", "exp"],
        "decay_alpha": [1],
        "random_seed": [14],
    }

    requirements = ["numpy", "scipy"]

    def download(self, raw_data_dir):
        pass

    def preprocess_and_save(self, raw_data_dir, data_dir):
        W = uniform(self.d, self.rank, self.random_seed)
        if self.decay_function == "linear":
            S = 1 / (1 + self.decay_alpha * np.arange(self.rank))
        elif self.decay_function == "sqrt":
            S = 1 / (1 + np.sqrt(self.decay_alpha * np.arange(self.rank)))
        elif self.decay_function == "exp":
            S = np.exp(-self.decay_alpha * np.linspace(0, 1, self.rank))
        else:
            raise ValueError(f"Unknown decay function {self.decay_function}")
        V = uniform(self.n, self.rank, self.random_seed + 1)
        X = W @ np.diag(S) @ V.T
        np.save(os.path.join(data_dir, "data.npy"), X)
        np.save(os.path.join(data_dir, "gt.npy"), W)

    def load(self, data_dir):
        X = np.load(os.path.join(data_dir, "data.npy"))
        W = np.load(os.path.join(data_dir, "gt.npy"))
        return dict(X=X, W=W)
