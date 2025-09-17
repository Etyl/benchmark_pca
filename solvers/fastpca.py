import numpy as np

from benchmark_utils import constants, stiefel
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchopt.base import BaseSolver

# pseudo-code from https://arxiv.org/pdf/2108.12373
# perform surprinsingly bad (in theory, self.prof = False) compared to announced results, can't exclude an implementation error


class Solver(BaseSolver):
    name = "fastpca"

    parameters = {"step_size": [1, 0.7, 1e-2, 1e-3], "size": [10, 20, 40], "proj": [True, False]}

    requirements = ["scipy"]

    stopping_criterion = SufficientProgressCriterion(
        eps=constants.EPS, patience=constants.PATIENCE, strategy="callback"
    )

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

    def slow_pseudo_gradient(self, C, X):
        h1 = np.zeros_like(X)
        h2 = np.zeros_like(h1)
        h3 = np.zeros_like(h1)

        for i in range(X.shape[1]):
            xi = X[:, i]
            h1[:, i] = C @ xi
            h2[:, i] = -(xi.T @ C @ xi / np.linalg.norm(xi) ** 2) * xi
            for j in range(i):
                xj = X[:, j]
                h3[:, i] -= (xj.T @ C @ xi / np.linalg.norm(xj) ** 2) * xj
        return h1 + h2 + h3

    def run(self, callback):
        self.random_seed = callback.meta["idx_rep"]
        size = self.size
        full_data = self.X
        k = self.n_components

        generator = np.random.default_rng(self.random_seed)
        perm = generator.permutation(full_data.shape[0])
        n = full_data.shape[0] // size

        # Each "rank" gets its own data slice and variables
        Xs = np.stack(
            [full_data[perm][n * i : n * (i + 1)].T for i in range(size)]
        )  # shape: (size, d, n)
        d, n = Xs.shape[1], Xs.shape[2]

        # Initialization
        Cs = np.matmul(Xs, np.transpose(Xs, (0, 2, 1)))  # (size, d, d)
        Ws = np.stack(
            [stiefel.uniform(d, k, random_seed=self.random_seed) for _ in range(size)]
        )  # (size, d, k)
        ss = np.zeros_like(Ws)
        hs = np.zeros_like(Ws)
        for i in range(size):
            hs[i] = self.slow_pseudo_gradient(Cs[i], Ws[i])
        ss = hs.copy()

        self.components = Ws[0] / np.linalg.norm(
            Ws[0], axis=0, keepdims=True
        )  # use first rank for evaluation

        while callback():
            # Simulate Allreduce for W
            W_rcv = Ws.mean(axis=0)
            Ws = Ws / 2 + W_rcv / 2 + self.step_size * ss

            if self.proj:
                for i in range(size):
                    Ws[i] = np.linalg.qr(Ws[i], mode="reduced")[0]

            # Simulate Allreduce for s
            s_rcv = ss.mean(axis=0)
            hs_new = np.zeros_like(hs)
            for i in range(size):
                hs_new[i] = self.slow_pseudo_gradient(Cs[i], Ws[i])
            ss = ss / 2 + s_rcv / 2 + (hs_new - hs)
            hs = hs_new

            W_eval = Ws[0] / np.linalg.norm(Ws[0], axis=0, keepdims=True)
            self.components = W_eval

    def get_result(self):
        return dict(components=self.components)
