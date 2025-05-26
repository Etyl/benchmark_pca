from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np 
    from benchmark_utils import constants, stiefel

# Pseudo-code from https://arxiv.org/pdf/1905.12115

class Solver(BaseSolver):

    name = 'adaoja'

    parameters = {
        "random_seed" : [constants.RANDOM_SEED],
        "b0" : [1e-5],
        "batch_size" : [1, 10, 50]
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
        b = np.full(k, self.b0)

        indices = generator.integers(0, n, (n_iter, self.batch_size)) #TODO: Add online version

        for iter in range(n_iter):
            Xb = self.X[indices[iter]].T # minibatch of dim (d, self.batch_size)
            G = (1 / self.batch_size) * Xb @ Xb.T @ W
            b = np.sqrt(b**2 + np.linalg.norm(G, axis=0))
            W = W + G / b[None, :] 
            W, _ = np.linalg.qr(W, mode="reduced")
        
        self.components = W

    def get_result(self):
        return dict(components=self.components)
