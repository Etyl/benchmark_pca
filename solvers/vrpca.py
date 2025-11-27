import numpy as np

from benchmark_utils import linalg
from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchmark_utils import constants, stiefel


# Pseudo-code from https://proceedings.mlr.press/v48/shamira16.pdf
class Solver(BaseSolver):
    name = "vrpca"

    parameters = {
        "step_size": [1e-3, 1e-2],
        "batch_size": [50],
        "epoch_fraction": [0.01, 0.1, 0.5],
    }

    requirements = ["scipy"]

    stopping_criterion = SufficientProgressCriterion(
        eps=constants.EPS, patience=constants.PATIENCE, strategy="callback"
    )

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

    def run(self, callback):
        self.random_seed = callback.meta["idx_rep"]
        generator = np.random.default_rng(self.random_seed)
        n, d = self.X.shape
        k = self.n_components
        epoch_size = int(n * self.epoch_fraction // self.batch_size)
        assert epoch_size > 0, "epoch_fraction is too small"

        X = self.X.T

        # initialization
        W = stiefel.uniform(d, k, self.random_seed)
        self.components = W
        W_tilde = np.empty_like(W)
        U_tilde = np.empty_like(W)

        i = 0
        while callback():
            if i % epoch_size == 0:
                W_tilde = W
                U_tilde = X @ (X.T @ W_tilde) / n

            indices = generator.integers(0, n, self.batch_size)
            U, _, Vh = np.linalg.svd(
                W.T @ W_tilde, full_matrices=False, compute_uv=True
            )
            B = Vh.T @ U.T
            Xb = self.X[indices].T
            W_prime = W + self.step_size * (
                Xb @ (Xb.T @ W - Xb.T @ (W_tilde @ B)) / self.batch_size +
                U_tilde @ B
            )
            W = W_prime @ linalg.inv_squared_root(W_prime.T @ W_prime)
            self.components = W
            i += 1

    def get_result(self):
        return dict(components=self.components)
