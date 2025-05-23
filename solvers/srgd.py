from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np 
    import scipy.stats 
    from benchmark_utils import constants, stiefel

# Pseudo-code from https://arxiv.org/pdf/1111.5280

class Solver(BaseSolver):

    name = 'SRGD'

    parameters = {
        "step_size" : [1e-4, 1e-3, 1e-2, 1e-1],
        "random_seed" : [constants.RANDOM_SEED],
        "retraction" : ["QR", "cayley", "polar"] # exp is too long to run
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

        W = stiefel.uniform(d, k, self.random_seed)

        indices = generator.integers(0, n, n_iter) #TODO: Add online version

        for iter in range(n_iter):
            x = self.X[indices[iter]]
            G = - x[:, None] @ (x[None, :] @ W)
            W = stiefel.rgd_step(W, G, self.step_size, mode=self.retraction)
            
        self.components = W

    def get_result(self):
        return dict(components=self.components)
