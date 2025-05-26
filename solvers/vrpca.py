from benchmark_utils import linalg
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np 
    from benchmark_utils import constants, stiefel

# Pseudo-code from https://proceedings.mlr.press/v48/shamira16.pdf
class Solver(BaseSolver):

    name = 'vrpca'

    parameters = {
        "step_size" : [1e-4, 1e-3, 1e-2],
        "epoch_size" : [10, 100, 1000], # still way less than indicated in paper ~ n
        "random_seed" : [constants.RANDOM_SEED]
        }

    requirements = ["scipy"]

    sampling_strategy = "iteration"

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components
            

    def run(self, n_iter):
        generator = np.random.default_rng(self.random_seed)
        n, d = self.X.shape
        k = self.n_components

        X = self.X.T

        W_tilde = stiefel.uniform(d, k, self.random_seed)


        indices = generator.integers(0, n, (n_iter, self.epoch_size)) #TODO: Add online version

        for i in range(n_iter):
            U_tilde = X@(X.T@W_tilde)/n 
            W = W_tilde 
            for j in range(self.epoch_size):
                U, _, Vh = np.linalg.svd(W.T@W_tilde, full_matrices=False, compute_uv=True)
                B = Vh.T@U.T 
                x = self.X[indices[i,j]]
                W_prime = W + self.step_size*(x[:, None] @ (x[None, :] @ W - x[None, :] @ (W_tilde @ B)) + U_tilde @ B)
                W = W_prime @ linalg.inv_squared_root(W_prime.T@W_prime)
            W_tilde = W 

        self.components = W_tilde

    def get_result(self):
        return dict(components=self.components)
