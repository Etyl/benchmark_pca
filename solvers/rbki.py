from benchopt.stopping_criterion import NoCriterion
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from benchopt.utils import profile
    import numpy as np
    from benchmark_utils import constants

# Pseudo-code from https://arxiv.org/pdf/2306.12418
# Use of "Simple" version, TODO: add the extended one to gain 33% speed


class Solver(BaseSolver):
    name = "rbki"

    parameters = {
        "random_seed": [constants.RANDOM_SEED],
        "q": [6],
        "oversampling_ratio": [1, 1.5, 2, 3],
    }

    requirements = ["scipy"]

    stopping_criterion = NoCriterion(strategy="callback")

    def get_next(self, stop_val):
        return stop_val + 1

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

    @profile
    def run(self, callback):
        generator = np.random.default_rng(self.random_seed)
        n, d = self.X.shape
        q = self.q

        k = self.n_components // q

        if self.n_components % q != 0:
            print(
                f"n_iter ({q}) is not a factor of n_components ({self.n_components}), truncation will be performed"
            )
            k = k + 1

        k = int(self.oversampling_ratio * k)

        X = self.X.T
        Y_t = generator.normal(0, 1, (n, k))

        # First iteration
        Z_t = X @ Y_t
        Z_t, _ = np.linalg.qr(Z_t, mode="reduced")
        Y_t = X.T @ Z_t

        Y = Y_t.copy()
        Z = Z_t.copy()

        U, _, _ = np.linalg.svd(Y.T, full_matrices=False, compute_uv=True)
        self.components = (Z @ U)[:, : self.n_components]
        callback()

        for i in range(self.q - 1):
            Z_t = X @ Y_t
            Z_t = Z_t - Z @ (Z.T @ Z_t)
            Z_t = Z_t - Z @ (Z.T @ Z_t)
            Z_t, _ = np.linalg.qr(Z_t, mode="reduced")
            Y_t = X.T @ Z_t
            Z = np.concatenate([Z, Z_t], axis=1)  # if too slow preallocate Z and Y at the beggining
            Y = np.concatenate([Y, Y_t], axis=1)

            U, _, _ = np.linalg.svd(Y.T, full_matrices=False, compute_uv=True)
            self.components = (Z @ U)[:, : self.n_components]
            callback()

    def get_result(self):
        return dict(components=self.components)
