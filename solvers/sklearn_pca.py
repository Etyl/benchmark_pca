from benchopt import BaseSolver, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.decomposition import PCA

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA

class Solver(BaseSolver):

    name = 'Sklearn-PCA'

    parameters = {
        'svd_solver' : ["full", "covariance_eigh", "arpack", "randomized"]
    }

    requirements = ["scikit-learn"]

    sampling_strategy = "run_once"

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

    # Here we ignore n_iter because we just want runtime of the algorithm (no optim)
    def run(self, n_iter):
        pca_transform = PCA(n_components = self.n_components, 
                              svd_solver = self.svd_solver,
                              whiten=True) # normalize eigenvectors
        pca_transform.fit(self.X)
        self.components = pca_transform.components_.T

    def get_result(self):
        return dict(components=self.components)
