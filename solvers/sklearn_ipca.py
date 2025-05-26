from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import NoCriterion

with safe_import_context() as import_ctx:
    from sklearn.decomposition import IncrementalPCA
    from sklearn.utils import gen_batches

# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html

class Solver(BaseSolver):
    name = 'sklearn_ipca'

    parameters = {
        "batch_size" : [1, 10, 50]
    }

    requirements = ["scikit-learn"]

    stopping_criterion = NoCriterion(strategy="callback")

    def get_next(self, stop_val):
        return stop_val + 1
    
    def set_objective(self, X, n_components):
        self.X, self.n_components = X, n_components

    def pre_run_hook(self, callback):
        self.ipca_transform = IncrementalPCA(n_components=self.n_components, 
                                             batch_size=self.batch_size, 
                                             whiten=True) # normalize eigenvectors

    def run(self, callback):
        for batch_slice in gen_batches(self.X.shape[0], self.batch_size):
            self.ipca_transform.partial_fit(self.X[batch_slice])
            self.components = self.ipca_transform.components_.T
            callback()

    def get_result(self):
        return dict(components=self.components)

