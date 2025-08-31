from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from scipy.linalg import svd


class Solver(BaseSolver):
    name = "lapack-svd"

    parameters = {"solver": ["gesvd", "gesdd"]}

    requirements = ["scipy"]

    sampling_strategy = "run_once"

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

    def run(self, n_iter):
        U, _, _ = svd(self.X.T, full_matrices=False, lapack_driver=self.solver)
        self.components = U[:, : self.n_components]

    def get_result(self):
        return dict(components=self.components)
