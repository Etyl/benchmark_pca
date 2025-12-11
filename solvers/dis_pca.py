from numpy.lib.format import open_memmap
from benchopt import safe_import_context
from collections import defaultdict
import time

from benchmark_utils.mpi_solver import DistributedMPISolver

with safe_import_context() as import_ctx:
    from mpi4py import MPI
    import numpy as np
    from sklearn.utils.extmath import randomized_svd

# https://proceedings.neurips.cc/paper_files/paper/2014/file/e968f1646c1c6c35422b64c0934772a4-Paper.pdf


class Solver(DistributedMPISolver):
    name = "dis-pca"

    parameters = {
        "n_workers": [4, 16],
    }

    requirements = ["numpy", "mpi4py", "scikit-learn"]

    sampling_strategy = "run_once"

    @classmethod
    def init_worker(cls, args, comm, rank, world_size):
        """
        Initialize worker, load data, and perform Global Mean Centering.
        (PCA requires centered data; we handle this once at initialization).
        """
        # 1. Load Data Slice
        X_mmap = open_memmap(args.data_path, mode='c')
        n_samples, n_features = X_mmap.shape

        chunk_size = n_samples // world_size
        start = rank * chunk_size
        end = start + chunk_size if rank != world_size - 1 else n_samples

        X_local = np.array(X_mmap[start:end])

        local_sum = np.sum(X_local, axis=0)
        global_sum = np.zeros_like(local_sum)

        comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        global_mean = global_sum / n_samples

        X_local_centered = X_local - global_mean

        return X_local_centered

    @classmethod
    def worker_run(cls, n_iter, X_local, args, comm, rank, world_size):
        n_features = X_local.shape[1]
        k = args.n_components
        logs = defaultdict(list)

        # Initialize output container
        W = np.zeros((n_features, k))

        # Local PCA
        t_start = time.perf_counter()
        _, _, Vt = np.linalg.svd(X_local, full_matrices=False)
        local_eigenvectors = Vt[:k, :].T.copy()
        logs['compute_time'].append(time.perf_counter() - t_start)

        # Communication
        t_start = time.perf_counter()

        if rank == 0:
            gathered_Vs = np.empty(
                (world_size, n_features, k),
                dtype=local_eigenvectors.dtype
            )
        else:
            gathered_Vs = None

        comm.Gather(local_eigenvectors, gathered_Vs, root=0)

        logs['comm_time'].append(time.perf_counter() - t_start)

        # Global Aggregation
        if rank == 0:
            t_start = time.perf_counter()

            # Stack: (n_features, world_size * k)
            V_stacked = np.hstack([gathered_Vs[w] for w in range(world_size)])

            U_global, _, _ = randomized_svd(
                V_stacked,
                n_components=k,
                n_oversamples=10,
            )
            W = np.ascontiguousarray(U_global)

            # Sign determinism
            signs = np.sign(W[0, :])
            signs[signs == 0] = 1.0
            W *= signs[None, :]

            logs['update_time'].append(time.perf_counter() - t_start)

        logs['sample_time'].append(0.0)
        logs['project_time'].append(0.0)

        return W, dict(logs)

    def get_result(self):
        W, logs = self.components
        return dict(components=W, logs=logs)


if __name__ == "__main__":
    Solver.entry_point()
