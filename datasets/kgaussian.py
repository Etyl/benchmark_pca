import os 
from benchopt import BaseDataset, safe_import_context
from benchopt.config import get_data_path

with safe_import_context() as import_ctx:
    import numpy as np
    import scipy.stats
    from benchmark_utils.data import DiskDataset
    from benchmark_utils.stiefel import uniform

def generate_gaussian(n_samples, n_features, rank, random_seed):
        generator = np.random.default_rng(random_seed)
        ortho_generator = scipy.stats.ortho_group(max(rank, n_features), generator)
        W = ortho_generator.rvs()[:n_features, :rank]
        C = generator.normal(size=(rank, n_samples))
        X = W @ C
        return X

class Dataset(DiskDataset):
    # data which is not low-rank, but which is really close to
    name = "kgaussian"

    parameters = {
        'n, d': [
            (100, 50),
            (500, 20),
        ],
        'rank' : [10], # "rank"
        'random_seed': [2112],
    }

    requirements = ["numpy", "scipy"]

    def download(self, raw_data_dir):
        pass

    def preprocess_and_save(self, raw_data_dir, data_dir):
        generator = np.random.default_rng(self.random_seed)
        W = uniform(self.d, self.rank, self.random_seed)
        C = generator.normal(size=(self.rank, self.n))
        X = W @ C
        np.save(os.path.join(data_dir, "data.npy"), X)

    def load(self, data_dir):
         X = np.load(os.path.join(data_dir, "data.npy"))
         return X