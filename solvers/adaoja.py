import numpy as np

from benchopt import BaseSolver
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchmark_utils import constants
from benchmark_utils import stiefel

# Pseudo-code from https://arxiv.org/pdf/1905.12115


class Solver(BaseSolver):
    name = "adaoja"

    parameters = {
        "b0": [1e-5],
        "batch_size": [32],
    }

    requirements = ["scipy"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    def set_objective(self, n, d, X_path, n_components):
        self.X = np.load(X_path)
        self.n_components = n_components

    def run(self, n_iter):
        self.random_seed = np.random.randint(10000)
        generator = np.random.default_rng(self.random_seed)

        n, d = self.X.shape
        k = self.n_components

        W = stiefel.uniform(d, k, self.random_seed)
        self.components = W
        b = np.full(k, self.b0)

        for _ in range(n_iter):
            indices = generator.integers(0, n, self.batch_size)
            Xb = self.X[indices].T  # minibatch of dim (d, self.batch_size)
            G = (1 / self.batch_size) * Xb @ (Xb.T @ W)
            b = np.sqrt(b**2 + np.linalg.vector_norm(G, axis=0) ** 2)
            W = W + G / b[None, :]
            W, _ = np.linalg.qr(W, mode="reduced")
            self.components = W

    def get_result(self):
        return dict(components=self.components)
