from sklearn.utils.extmath import randomized_svd

from benchopt import BaseSolver


class Solver(BaseSolver):
    name = "randomized-pca"

    parameters = {
        "n_oversamples": [10],
        "n_iter": [0, 4, 7]
    }

    requirements = ["scikit-learn"]

    sampling_strategy = "run_once"

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

    def run(self, n_iter):
        U, _, _ = randomized_svd(
            self.X.T,
            n_components=self.n_components,
            n_oversamples=self.n_oversamples,
            n_iter=self.n_iter,
            random_state=None,
        )
        # no access to rep_idx
        self.components = U

    def get_result(self):
        return dict(components=self.components)
