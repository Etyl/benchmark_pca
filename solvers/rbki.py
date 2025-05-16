from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np 
    import scipy.stats 
    from benchmark_utils import constants

# Pseudo-code from https://arxiv.org/pdf/2306.12418
# Use of "Simple" version, TODO: add the extended one

#TODO: (iteration sampling with custom get_next() not to grow too fast)
class Solver(BaseSolver):

    name = 'Oja'

    parameters = {
        "block_size": [10, 30, 50, 70],
        "n_iter": [1,2,4],
        "random_seed" : [constants.RANDOM_SEED]
        }

    requirements = ["scipy"]

    sampling_strategy = "iteration"

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components
            

    def run(self, n_iter):
        generator = np.random.default_rng(self.random_seed)
        X = self.X 
        n, d = X.shape
        k = self.n_components

        Y = generator.normal(0, 1, (n, self.block_size)) 

        Z_list = []
        Y_list = []
        for iter in range(self.n_iter):
            Z = X.T@Y
            new_Z = Z.copy()
            for old_Z in Z_list:
                new_Z = new_Z - old_Z @ (old_Z.T @ Z)
            for old_Z in Z_list:
                new_Z = new_Z - old_Z @ (old_Z.T @ Z)
            Z = new_Z 
            Z, _ = np.linalg.qr(Z, mode="reduced") 
            Z_list.append(Z)
            Y = X@Z
            Y_list.append(Y)
        
        U, S, Vh = np.linalg.svd(np.concat(Y_list, axis=1).T)
        self.components = np.concat(Z_list, axis=1) @ U #of dim (d, block_size)

    def get_result(self):
        return dict(components=self.components)
