from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import time

class Objective(BaseObjective):

    name = "PCA"

    url = "https://github.com/MortimerTP/benchmark_distributed_pca"

    parameters = {
        'n_components' : [2,10,20]
    }

    requirements = ["numpy"]

    min_benchopt_version = "1.5" # To check

    def set_data(self, X):
        self.X = X 
        self.X_norm = np.linalg.norm(X, 'fro')

    def evaluate_result(self, components):
        print("begin ortho : ", time.perf_counter())
        
        orthogonality_gap = np.max(np.abs(components @ components.T - np.eye(components.shape[0])).flatten())
        print("end ortho - begin proj", time.perf_counter())
        energy_gap = np.linalg.norm(self.X - self.X@(components.T@components), 'fro')
        energy_gap = energy_gap / self.X_norm
        print("end proj",  time.perf_counter())
        return dict(value=energy_gap, orthogonality_gap=orthogonality_gap)  

    def get_one_result(self):
        return dict(components=np.zeros(self.n_components, self.X.shape[1]))

    def get_objective(self):
        return dict(
            X=self.X,
            n_components=self.n_components
        )
