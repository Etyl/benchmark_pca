from benchopt import BaseDataset, safe_import_context


# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import numpy as np


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "OneSpike"

    # List of parameters to generate the datasets. The benchmark will consider
    # the cross product for each key in the dictionary.
    # Any parameters 'param' defined here is available as `self.param`.
    parameters = {
        'n_samples, n_features': [
            (100, 50),
            (500, 20),
        ],
        'sigma' : [0.2],
        'random_state': [0],
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = ["numpy"]

    ##TODO: decentralized data generation (and only pass a reference ?)
    def get_data(self):
        # The return arguments of this function are passed as keyword arguments
        # to `Objective.set_data`. This defines the benchmark's
        # API to pass data. It is customizable for each benchmark.

        # Generate pseudorandom data using `numpy`.
        rng = np.random.RandomState(self.random_state)
        
        ## random sampling on hypersphere
        w = rng.normal(size=(self.n_features))
        w = w / np.linalg.norm(w)

        r = rng.normal(size=(self.n_samples))
        eps = rng.normal(scale=self.sigma, size=(self.n_samples, self.n_features))

        X = r[:, ...] * w[..., :] + eps
        y = w

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(X=X, y=y)
