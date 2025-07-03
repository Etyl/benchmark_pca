from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from benchopt.stopping_criterion import SufficientProgressCriterion
    from benchmark_utils import constants, stiefel

# Pseudo-code from https://proceedings.neurips.cc/paper_files/paper/2019/file/38faae069a1371784081ea9ad9b279d0-Paper.pdf
# in which convention X.shape = (k,d)
# TODO: Add Warm Start version


class Solver(BaseSolver):
    name = "krasulina"

    parameters = {
        "step_size": [1e-3, 1e-1],
        "random_seed": [constants.RANDOM_SEED],
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
        generator = np.random.default_rng(self.random_seed)
        n, d = self.X.shape
        k = self.n_components

        W = stiefel.uniform(d, k, self.random_seed)
        self.components = W

        while callback():
            indices = generator.integers(0, n, self.batch_size)
            Xb = self.X[indices].T  # minibatch of dim (d, self.batch_size)
            wx = W.T @ Xb  # micro optim to save kdb iterations
            W = W + self.step_size * (Xb - W @ (wx)) @ wx.T / self.batch_size
            W, _ = np.linalg.qr(
                W, mode="reduced"
            )  # TODO: Add "stabilized" SVD based orthogonalization
            self.components = W

    def get_result(self):
        return dict(components=self.components)
