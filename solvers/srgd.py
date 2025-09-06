from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.stopping_criterion import NoCriterion
    from benchmark_utils import stiefel

# Pseudo-code from https://arxiv.org/pdf/1111.5280


class Solver(BaseSolver):
    name = "srgd"

    parameters = {
        "step_size": [1e-3, 1e-2],
        "batch_size": [10],
        "retraction": [
            "QR",
            "polar",
            "exp",
            "cayley",
        ],  # exp is too long to run and polar is unstable
    }

    requirements = ["scipy"]

    # stopping_criterion = SufficientProgressCriterion(
    #     eps=constants.EPS, patience=constants.PATIENCE, strategy="callback"
    # )

    stopping_criterion = NoCriterion(strategy="callback")

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

    def run(self, callback):
        self.random_seed = callback.meta["idx_rep"]
        generator = np.random.default_rng(self.random_seed)
        X = self.X
        n, d = X.shape
        k = self.n_components

        W = stiefel.uniform(d, k, self.random_seed)
        self.components = W

        while callback():
            indices = generator.integers(0, n, self.batch_size)
            Xb = self.X[indices].T
            G = -Xb @ (Xb.T @ W) / self.batch_size
            W = stiefel.rgd_step(W, G, self.step_size, mode=self.retraction)
            self.components = W

    def get_result(self):
        return dict(components=self.components)
