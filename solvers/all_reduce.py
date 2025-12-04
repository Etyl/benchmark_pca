from numpy.lib.format import open_memmap
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchmark_utils.mpi_solver import DistributedMPISolver
import benchmark_utils.stiefel as stiefel

with safe_import_context() as import_ctx:
    from mpi4py import MPI
    import numpy as np


class Solver(DistributedMPISolver):
    name = "all-reduce"

    parameters = {
        "n_workers": [4, 16],
        "batch_size": [128],
        "b0": [1e-5],
    }

    requirements = ["numpy", "mpi4py"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    @classmethod
    def init_worker(cls, args, comm, rank, world_size):
        """
        Initialize the worker environment and load data.
        Returns the local data tensor (X_local).
        """
        # Load Data Slice
        X_mmap = open_memmap(args.data_path, mode='c')
        n_samples, n_features = X_mmap.shape

        chunk_size = n_samples // world_size
        start = rank * chunk_size
        end = start + chunk_size if rank != world_size - 1 else n_samples

        return X_mmap[start:end]

    @classmethod
    def worker_run(cls, n_iter, X_local, args, comm, rank, world_size):
        """
        The core optimization logic.
        """
        n_features = X_local.shape[1]

        # Re-init weights for every run
        W = stiefel.uniform(n_features, args.n_components)
        b = np.full((args.n_components,), args.b0)

        G_local = np.zeros((n_features, args.n_components))

        for _ in range(n_iter):
            indices = np.random.randint(0, len(X_local), (args.batch_size,))
            X_batch = X_local[indices]
            np.matmul(X_batch.T, X_batch @ W, out=G_local)

            # Sum gradients across all workers
            comm.Allreduce(MPI.IN_PLACE, G_local, op=MPI.SUM)
            G_local /= args.batch_size

            b = np.sqrt(b**2 + np.linalg.vector_norm(G_local, axis=0) ** 2)
            W += G_local / b[None, :]
            W, _ = np.linalg.qr(W, mode='reduced')

        return W


if __name__ == "__main__":
    Solver.entry_point()
