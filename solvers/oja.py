from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np 
    from benchmark_utils import constants, stiefel

# Pseudo-code from https://arxiv.org/pdf/1905.12115

class Solver(BaseSolver):

    name = 'oja'

    parameters = {
        "step_size" : [1e-4, 1e-3, 1e-2, 1e-1],
        "random_seed" : [constants.RANDOM_SEED],
        "batch_size" : [1, 10]
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

        W = stiefel.uniform(d, k, self.random_seed)

        indices = generator.integers(0, n, (n_iter, self.batch_size)) #TODO: Add online version

        for iter in range(n_iter):
            Xb = self.X[indices[iter]].T 
            W = W + self.step_size * Xb @ (Xb.T @ W)
            W, _ = np.linalg.qr(W, mode="reduced") #TODO: Add "stabilized" SVD based orthogonalization
    
        self.components = W

    def get_result(self):
        return dict(components=self.components)
