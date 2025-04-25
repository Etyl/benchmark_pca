from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np

class Objective(BaseObjective):

    name = "PCA"

    url = "https://github.com/MortimerTP/benchmark_distributed_pca"

    parameters = {
        'n_components' : [2,10,20]
    }

    requirements = ["numpy"]

    min_benchopt_version = "1.5" ## To check

    def set_data(self, X):
        self.X = X 

    def evaluate_result(self, pca, var_ratio):
        unexplained_var = 1 - var_ratio.sum()
        return dict(value=unexplained_var)  

    def get_one_result(self):
        return dict(pca=np.zeros(self.n_components, self.X.shape[1]), var_ratio=np.zeros(self.n_components))

    def get_objective(self):
        return dict(
            X=self.X,
            n_components=self.n_components
        )
