import os 
from benchopt import BaseDataset, safe_import_context
from benchmark_utils.data import is_data_valid, mark_data_status
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    import numpy as np
    import scipy.stats

def generate_gaussian(n_samples, n_features, rank, random_seed):
        generator = np.random.default_rng(random_seed)
        ortho_generator = scipy.stats.ortho_group(max(rank, n_features), generator)
        W = ortho_generator.rvs()[:n_features, :rank]
        C = generator.normal(size=(rank, n_samples))
        X = W @ C
        return X

class Dataset(BaseDataset):
    # data which is not low-rank, but which is really close to
    name = "k-Gaussian"

    parameters = {
        'n_samples, n_features': [
            (100, 50),
            (500, 20),
        ],
        'rank' : [10], # "rank"
        'random_seed': [2112],
    }

    requirements = ["numpy", "scipy"]

    def get_data(self):
        base_path = get_data_path()
        data_dir = os.path.join(base_path, "gaussian")
        data_path = os.path.join(data_dir, "data.npy")
        status_path = os.path.join(data_dir, "status.json")

        if os.path.exists(data_path) and is_data_valid(status_path):
            print("Using cached Gaussian synthetic data.")
            X=np.load(data_path)

        else: 
            print("Generating synthetic Gaussian data...")
            mark_data_status(status_path, "incomplete")

            X = generate_gaussian(self.n_samples, self.n_features, self.rank, self.random_seed)

            os.makedirs(data_dir, exist_ok=True)
            np.save(data_path, X)
            mark_data_status(status_path, "complete")

        return dict(X=X)