from numpy.lib.format import open_memmap
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
from collections import defaultdict
import time

from benchmark_utils.mpi_solver import DistributedMPISolver
import benchmark_utils.stiefel as stiefel

with safe_import_context() as import_ctx:
    from mpi4py import MPI
    import numpy as np


class Solver(DistributedMPISolver):
    name = "reduce-cast"

    parameters = {
        "n_workers": [1, 4, 16],
        "batch_size": [64],
        "b0": [1e-5],
        "project_every": [3],
    }

    requirements = ["numpy", "mpi4py"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    @classmethod
    def init_worker(cls, args, comm, rank, world_size):
        """
        Initialize the worker environment, clear logs, and load data.
        Returns the local data tensor (X_local).
        """
        # Load Data Slice
        X_mmap = open_memmap(args.data_path, mode='c')
        n_samples, n_features = X_mmap.shape

        chunk_size = n_samples // world_size
        start = rank * chunk_size
        end = start + chunk_size if rank != world_size - 1 else n_samples

        X_local = np.array(X_mmap[start:end])

        return X_local

    @classmethod
    def worker_run(cls, n_iter, X_local, args, comm, rank, world_size):
        n_features = X_local.shape[1]
        logs = defaultdict(list)

        # Re-init weights for every run
        W = stiefel.uniform(n_features, args.n_components, random_seed=0)
        b = np.full((args.n_components,), args.b0)

        G_local = np.zeros((n_features, args.n_components))
        G_global = np.zeros((n_features, args.n_components))

        if args.batch_size % world_size != 0:
            raise ValueError(
                "Batch size must be divisible by number of workers "
                f"({args.batch_size} % {world_size} != 0)"
            )
        local_batch_size = args.batch_size // world_size

        for k in range(n_iter):
            # Sampling
            t_start = time.perf_counter()
            indices = np.random.randint(0, len(X_local), (local_batch_size,))
            X_batch = X_local[indices]
            logs['sample_time'].append(time.perf_counter() - t_start)

            # Local Computation (Gradient)
            t_start = time.perf_counter()
            np.matmul(X_batch.T, X_batch @ W, out=G_local)
            logs['compute_time'].append(time.perf_counter() - t_start)

            # Gather gradients to Rank 0
            t_start = time.perf_counter()
            comm.Reduce(G_local, G_global, op=MPI.SUM, root=0)
            logs['comm_time'].append(time.perf_counter() - t_start)

            # 4. Global Update & Orthogonalization (Rank 0 only)
            if rank == 0:
                # Update
                t_start = time.perf_counter()
                G_global /= args.batch_size
                b = np.sqrt(b**2 + np.linalg.vector_norm(G_global, axis=0)**2)
                W += G_global / b[None, :]
                logs['update_time'].append(time.perf_counter() - t_start)

                # Orthogonalization (QR)
                t_start = time.perf_counter()
                if (k + 1) % args.project_every == 0 or k == n_iter - 1:
                    W, _ = np.linalg.qr(W, mode='reduced')
                    signs = np.sign(W[0, :])
                    signs[signs == 0] = 1.0
                    W *= signs[None, :]
                logs['project_time'].append(time.perf_counter() - t_start)

            # Broadcast new W to all workers
            t_start = time.perf_counter()
            comm.Bcast(W, root=0)
            logs['comm_time'][-1] += (time.perf_counter() - t_start)

        return W, dict(logs)

    def get_result(self):
        # Unpack the tuple (W, logs) returned by the worker
        W, logs = self.components
        return dict(components=W, logs=logs)


if __name__ == "__main__":
    Solver.entry_point()
