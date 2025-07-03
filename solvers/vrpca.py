from benchmark_utils import linalg
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.stopping_criterion import SufficientProgressCriterion
    from benchmark_utils import constants, stiefel


# Pseudo-code from https://proceedings.mlr.press/v48/shamira16.pdf
class Solver(BaseSolver):
    name = "vrpca"

    parameters = {
        "step_size": [1e-3, 1e-2, 1e-1, 1],
        "batch_size": [1, 10],
        "epoch_size": [10, 100, 1000],  # still way less than indicated in paper ~ n
        "random_seed": [constants.RANDOM_SEED],
    }

    requirements = ["scipy"]

    stopping_criterion = SufficientProgressCriterion(
        eps=constants.EPS, patience=constants.PATIENCE, strategy="callback"
    )

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

    def run(self, callback):
        generator = np.random.default_rng(self.random_seed)
        n, d = self.X.shape
        k = self.n_components

        X = self.X.T

        # initialization
        W = stiefel.uniform(d, k, self.random_seed)
        self.components = W
        W_tilde = np.empty_like(W)
        U_tilde = np.empty_like(W)

        i = 0
        while callback():
            if i % self.epoch_size == 0:
                W_tilde = W
                U_tilde = X @ (X.T @ W_tilde) / n

            indices = generator.integers(0, n, self.batch_size)
            U, _, Vh = np.linalg.svd(W.T @ W_tilde, full_matrices=False, compute_uv=True)
            B = Vh.T @ U.T
            Xb = self.X[indices].T
            W_prime = W + self.step_size * (Xb @ (Xb.T @ W - Xb.T @ (W_tilde @ B)) + U_tilde @ B)
            W = W_prime @ linalg.inv_squared_root(W_prime.T @ W_prime)
            self.components = W_tilde
            i += 1

    def get_result(self):
        return dict(components=self.components)
