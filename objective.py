from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class Objective(BaseObjective):
    name = "pca"

    parameters = {"n_components": [20]}

    requirements = ["numpy"]

    min_benchopt_version = "1.5"  # To check

    def set_data(self, n, d, X_path, W=None):
        self.n = n
        self.d = d
        self.X_path = X_path
        if W is not None:
            self.W = W

    def evaluate_result(self, components):
        if (
            components.shape[0] != self.n or
            components.shape[1] != self.n_components
        ):
            raise ValueError(
                f"components should be of shape "
                f"({self.n},{self.n_components}), "
                f"current shape {components.shape}"
            )
        ortho_diagnostic = np.max(
            np.abs(components.T @ components - np.eye(self.n_components))
        ).flatten()

        X = np.load(self.X_path, mmap_mode="r")

        x_norm = 0
        xc_norm = 0
        for i in range(self.n):
            x = X[i, :]
            x_norm += np.sum(x**2)
            xc_norm += np.sum((x[None, :] @ components)**2)

        unexplained_var = 1 - xc_norm / x_norm

        return dict(value=unexplained_var, ortho_diagnostic=ortho_diagnostic)

    def get_one_result(self):
        return dict(components=np.zeros((self.n, self.n_components)))

    def get_objective(self):
        return dict(
            n=self.n,
            d=self.d,
            X_path=self.X_path,
            n_components=self.n_components
        )
