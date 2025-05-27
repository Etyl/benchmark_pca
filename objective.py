from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

class Objective(BaseObjective):

    name = "pca"

    url = "https://github.com/MortimerTP/benchmark_distributed_pca"

    parameters = {
        'n_components' : [2,12,24]
    }

    requirements = ["numpy"]

    min_benchopt_version = "1.5" # To check

    def set_data(self, X):
        self.X = X 
        self.X_norm = np.linalg.norm(X, 'fro')

    def evaluate_result(self, components):
        if components.shape[0] != self.X.shape[1] or components.shape[1] != self.n_components:
            raise ValueError(f"components should be of shape ({self.X.shape[1]},{self.n_components}), current shape {components.shape}")
        ortho_diagnostic = np.max(np.abs(components.T @ components - np.eye(self.n_components)).flatten())
        unexplained_var = 1 - np.linalg.norm(self.X @ components, "fro") / self.X_norm
        return dict(value=unexplained_var, ortho_diagnostic=ortho_diagnostic)  

    def get_one_result(self):
        return dict(components=np.zeros((self.X.shape[1], self.n_components)))

    def get_objective(self):
        return dict(
            X=self.X,
            n_components=self.n_components
        )
