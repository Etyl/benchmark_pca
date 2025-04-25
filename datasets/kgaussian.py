from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):

    name = "k-Gaussian"

    parameters = {
        'n_samples, n_features': [
            (100, 50),
            (500, 20),
        ],
        'k' : [10], ## nb of non-zero components
        'random_state': [2112],
    }

    requirements = ["numpy"]

    def get_data(self):

        # Generate pseudorandom data using `numpy`.
        rng = np.random.RandomState(self.random_state)

        ## TODO: Generate a sequence of orthogonal vectors instead
        ## random sampling on hypersphere
        vec = rng.normal(size=(self.k, self.n_features))
        vec = vec / np.linalg.norm(vec, axis=1)[:,None]
        
        values = np.abs(rng.normal(size=(self.k)))

        X = rng.normal(size=(self.n_samples, self.k))

        X = X @ (values[:, None] * vec)

        return dict(X=X)
