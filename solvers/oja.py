from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.stopping_criterion import SufficientProgressCriterion
    from benchmark_utils import stiefel
    from benchmark_utils import constants

# Pseudo-code from https://arxiv.org/pdf/1905.12115


class Solver(BaseSolver):
    name = "oja"

    parameters = {
        "step_size": [1e-2, 1e-3],
        "batch_size": [1, 10],
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

        W = stiefel.uniform(d, k, self.random_seed)
        step_size = self.step_size
        self.components = W
        while callback():
            indices = generator.integers(0, n, self.batch_size)
            Xb = self.X[indices].T  # minibatch of dim (d, self.batch_size)
            W = W + step_size * Xb @ (Xb.T @ W) / self.batch_size
            W, _ = np.linalg.qr(W, mode="reduced")
            self.components = W

    def get_result(self):
        return dict(components=self.components)
