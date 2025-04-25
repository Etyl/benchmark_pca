from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import StoppingCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from sklearn.decomposition import IncrementalPCA
    from sklearn.utils import gen_batches

class FiniteTimeStoppingCriterion(StoppingCriterion):
    def check_convergence(self, objective_list):
        if not hasattr(self, "n_iter") or self.n_iter is None : 
            raise Exception("n_iter must be set in the solver")
        current_iter = len(objective_list)
        return False, current_iter / self.n_iter ## if current_iter == len(objective_list), can also rely directly on it


class Solver(BaseSolver):
    name = 'Sklearn-IPCA'

    parameters = {
        "batch_size" : [1, 10, 50]
    }

    requirements = ["scikit-learn"]

    stopping_criterion = FiniteTimeStoppingCriterion(strategy="callback")

    def get_next(self, stop_val):
        return stop_val + 1
    
    def set_objective(self, X, n_components):
        self.X, self.n_components = X, n_components

    def pre_run_hook(self, callback):
        n_batch = int(np.ceil(self.X.shape[0] / self.batch_size)) 
        callback.stopping_criterion.n_iter = n_batch

    def run(self, callback):
        ipca_transform = IncrementalPCA(n_components=self.n_components, batch_size=self.batch_size)
        for batch_slice in gen_batches(self.X.shape[0], self.batch_size):
            ipca_transform.partial_fit(self.X[batch_slice])
            self.pca = ipca_transform.components_
            self.var_ratio = ipca_transform.explained_variance_ratio_
            callback()

    def get_result(self):
        return dict(pca=self.pca, var_ratio=self.var_ratio)

