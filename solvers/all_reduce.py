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

    # batch_size now represents the GLOBAL batch size (e.g., 32)
    parameters = {
        "n_workers": [4, 16],
        "batch_size": [512],
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

        # --- 1. Calculate Local Batch Size ---
        # We split the global batch_size among workers.
        # e.g., if global batch_size=32 and 4 workers, local=8.
        global_batch_size = args.batch_size
        local_batch_size = global_batch_size // world_size
        if local_batch_size < 1:
            local_batch_size = 1
            if rank == 0:
                print(
                    f"Warning: batch_size {global_batch_size} "
                    f"< n_workers {world_size}. Using local_batch_size=1."
                )

        # Re-init weights for every run
        W = stiefel.uniform(
            n_features, args.n_components,
            random_seed=rank
        )
        b = np.full((args.n_components,), args.b0)

        G_local = np.zeros((n_features, args.n_components))

        # Optimization Loop
        for i in range(n_iter):

            # Sample local mini-batch
            indices = np.random.randint(0, len(X_local), (local_batch_size,))
            X_batch = X_local[indices]

            # Compute Sum of Gradients on local batch
            # G = sum(x * x^T * W)
            np.matmul(X_batch.T, X_batch @ W, out=G_local)

            # Sum gradients across all workers
            # Result in G is sum over GLOBAL batch
            comm.Allreduce(MPI.IN_PLACE, G_local, op=MPI.SUM)

            # --- 2. Scale by Global Batch Size ---
            # Compute the Mean Gradient: Sum / Global_Count
            # This matches adaoja: G = (1/B) * sum(grads)
            G_local /= global_batch_size

            b = np.sqrt(b**2 + np.linalg.vector_norm(G_local, axis=0) ** 2)

            W += G_local / b[None, :]
            W, _ = np.linalg.qr(W, mode='reduced')

        return W


if __name__ == "__main__":
    Solver.entry_point()
