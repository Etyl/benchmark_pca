from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np 
    import scipy.stats 
    from benchmark_utils import constants

# Pseudo-code from https://arxiv.org/pdf/1905.12115

class Solver(BaseSolver):

    name = 'Oja'

    parameters = {
        "step_size" : [1e-4, 1e-3, 1e-2, 1e-1],
        "random_seed" : [constants.RANDOM_SEED]
        }

    requirements = ["scipy"]

    sampling_strategy = "iteration"

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components
            

    def run(self, n_iter):
        generator = np.random.default_rng(self.random_seed)
        X = self.X 
        n, d = X.shape
        k = self.n_components

        ortho_generator = scipy.stats.ortho_group(max(k, d), generator)
        W = ortho_generator.rvs()[:d, :k]

        indices = generator.integers(0, n, n_iter) #TODO: Add online version

        for iter in range(n_iter):
            x = self.X[indices[iter]]
            W = W + self.step_size * x[:, None] @ (x[None, :] @ W)
            W, _ = np.linalg.qr(W, mode="reduced") #TODO: Add "stabilized" SVD based orthogonalization
    
        self.components = W

    def get_result(self):
        return dict(components=self.components)
