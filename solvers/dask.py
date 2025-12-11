import os
import atexit
from benchopt import BaseSolver, safe_import_context
from benchmark_utils import ACTIVE_SOLVERS

with safe_import_context() as import_ctx:
    import dask.array as da
    from dask_ml.decomposition import PCA
    from dask_jobqueue import SLURMCluster
    from dask.distributed import Client
    import numpy as np


class Solver(BaseSolver):
    name = "dask"

    parameters = {
        "n_workers": [1, 4, 16],
    }

    requirements = ["dask", "dask-ml", "dask-jobqueue"]

    sampling_strategy = "run_once"

    def set_objective(self, n, d, X_path, n_components):
        data_local = np.load(X_path)
        self.X = da.from_array(data_local, chunks="128MiB")
        self.n_components = n_components

    def warm_up(self):
        for solver in ACTIVE_SOLVERS:
            solver.cleanup()
        ACTIVE_SOLVERS.clear()

        cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        self.cluster = SLURMCluster(
            cores=cores,
            memory="8GB" if self.n_workers >= 4 else "16GB",
            processes=1,
            walltime="00:30:00",
        )
        self.cluster.scale(jobs=self.n_workers)
        self.client = Client(self.cluster)

        print(f"[{self.name}] Waiting for {self.n_workers} workers...")
        self.client.wait_for_workers(self.n_workers)

        ACTIVE_SOLVERS.append(self)
        atexit.register(self.cleanup)

    def run(self, n_iter):
        # svd_solver='randomized' is generally faster/stable for dask
        pca = PCA(n_components=self.n_components, svd_solver='randomized')
        pca.fit(self.X)
        self.components_ = pca.components_.T

    def get_result(self):
        return {"components": self.components_, "logs": {}}

    def cleanup(self):
        if self in ACTIVE_SOLVERS:
            ACTIVE_SOLVERS.remove(self)
        atexit.unregister(self.cleanup)

        if hasattr(self, 'client') and self.client:
            self.client.close()
        if hasattr(self, 'cluster') and self.cluster:
            self.cluster.close()

    def __del__(self):
        self.cleanup()
