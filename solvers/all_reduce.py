import numpy as np

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchmark_utils.all_reduce import worker_oja_step

with safe_import_context() as import_ctx:
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client, wait
    import dask.array as da
    from benchmark_utils import stiefel


class Solver(BaseSolver):
    name = "all-reduce"

    parameters = {
        "step_size": [1e-2],
        "batch_size": [10],
        "n_workers": [4],
        # SLURM generic configuration (adjust queues/time as needed)
        "cores_per_worker": [6],
        "memory_per_worker": ["6GB"],
        "walltime": ["00:30:00"]
    }

    requirements = ["numpy", "dask", "dask_jobqueue", "distributed"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="callback"
    )

    def set_objective(self, X, n_components):
        self.X = X
        self.n_components = n_components

        # 1. Setup SLURM Cluster (Moved to set_objective to exclude from timing)
        self.cluster = SLURMCluster(
            cores=self.cores_per_worker,
            memory=self.memory_per_worker,
            walltime=self.walltime,
            processes=1,  # 1 worker per job
        )

        self.cluster.scale(jobs=self.n_workers)
        self.client = Client(self.cluster)

        # Wait for workers to come online
        print("Waiting for workers...")
        self.client.wait_for_workers(n_workers=self.n_workers)

        # 2. Distribute Data (Moved to set_objective)
        # Scatter chunks of X to workers.
        X_splits = np.array_split(self.X, self.n_workers)

        # remote_X is a list of Futures pointing to data on workers
        self.remote_X = self.client.scatter(X_splits)
        print("Data distributed to workers.")

    def run(self, callback):
        # 3. Initialization
        n, d = self.X.shape
        k = self.n_components

        self.random_seed = callback.meta["idx_rep"]
        W = stiefel.uniform(d, k, self.random_seed)
        self.components = W

        iteration = 0

        try:
            while callback():
                # 4. Distributed Computation
                # Map the worker function to the distributed data chunks
                futures = [
                    self.client.submit(
                        worker_oja_step,
                        X_block=x_chunk,
                        W=W,
                        batch_size=self.batch_size,
                        seed=self.random_seed + iteration
                    )
                    for x_chunk in self.remote_X
                ]

                # 5. Aggregation (AllReduce equivalent)
                # Gather gradients back to the driver (Parameter Server)
                local_grads = self.client.gather(futures)

                # Average the gradients
                G_global = np.mean(local_grads, axis=0)

                # 6. Global Update & Retraction
                W = W + self.step_size * G_global
                W, _ = np.linalg.qr(W, mode="reduced")

                self.components = W
                iteration += 1

        finally:
            # Cleanup SLURM jobs at the end of the run
            self.client.close()
            self.cluster.close()

    def get_result(self):
        return dict(components=self.components)