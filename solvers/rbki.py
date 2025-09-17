from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

# Pseudo-code from https://arxiv.org/pdf/2306.12418
# Use of "Simple" version, TODO: add the extended one to gain 33% speed


class Solver(BaseSolver):
    name = "rbki"

    parameters = {
        "q": [1, 3, 6],
        "oversampling_ratio": [1, 1.5, 2, 3],
    }

    requirements = ["numpy"]

    sampling_strategy = "run_once"

    def get_next(self, stop_val):
        return stop_val + 1

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

    def run(self, n_iter):
        generator = np.random.default_rng()  # no reproducibility, as no access to rep_idx
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

    def get_result(self):
        return dict(components=self.components)
