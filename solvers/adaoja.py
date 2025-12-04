import numpy as np
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Import the new abstract base class
from benchmark_utils.single_solver import SingleNodeSolver

with safe_import_context() as import_ctx:
    from benchmark_utils import stiefel


class Solver(SingleNodeSolver):
    name = "adaoja"

    parameters = {
        "b0": [1e-5],
        "batch_size": [512],
    }

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    @classmethod
    def solve(cls, X, args):
        """
        Implementation of the AdaOja optimization logic.
        Called by the worker when a RUN command is received.
        """
        n, d = X.shape
        k = args.n_components

        # Random Initialization
        # A new seed is generated for each run to match benchopt's repetition behavior
        random_seed = np.random.randint(10000)
        generator = np.random.default_rng(random_seed)

        W = stiefel.uniform(d, k, random_seed)
        b = np.full(k, args.b0)

        # Optimization Loop
        for _ in range(args.n_iter):
            indices = generator.integers(0, n, args.batch_size)
            Xb = X[indices].T  # minibatch (d, batch_size)

            G = (1 / args.batch_size) * Xb @ (Xb.T @ W)

            b = np.sqrt(b**2 + np.linalg.norm(G, axis=0) ** 2)
            W = W + G / b[None, :]
            W, _ = np.linalg.qr(W, mode="reduced")

        return W


if __name__ == "__main__":
    Solver.entry_point()
