import os
import time
import json
import uuid  # Added for unique temp filenames
import sys   # Added for flushing stdout
import numpy as np
from numpy.lib.format import open_memmap
from benchopt import safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

from benchmark_utils.mpi_solver import DistributedMPISolver

with safe_import_context() as import_ctx:
    import torch
    from mpi4py import MPI


class Solver(DistributedMPISolver):
    name = "all-reduce"

    parameters = {
        "n_workers": [4],
        "batch_size": [32],
        "b0": [1e-5],
    }

    requirements = ["numpy", "torch", "mpi4py"]

    stopping_criterion = SufficientProgressCriterion(
        eps=1e-10, patience=3, strategy="iteration"
    )

    # Note: set_objective is inherited from DistributedMPISolver
    # It now includes the --overlap flag automatically.

    @classmethod
    def worker(cls, args):
        # --- 1. Init Environment & MPI ---
        # Prevent Thread Oversubscription
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", None)
        if slurm_cpus:
            torch.set_num_threads(int(slurm_cpus))
        else:
            torch.set_num_threads(4)  # Fallback

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # FIX: Robust Rank Detection
        # In some SLURM environments (especially with srun --overlap), mpi4py might
        # fail to bootstrap correctly and default to rank=0, world_size=1 for all tasks.
        if "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
        if "SLURM_NTASKS" in os.environ:
            world_size = int(os.environ["SLURM_NTASKS"])

        # Debug print to verify distinct ranks
        print(f"Worker initialized: Rank {rank}/{world_size}, PID {os.getpid()}")
        sys.stdout.flush()

        # Paths for coordination
        control_file = os.path.join(args.tmp_dir, "control.json")
        status_file = os.path.join(args.tmp_dir, "status_rank_0.txt")
        out_path = os.path.join(args.tmp_dir, "results.npy")

        # --- 2. Load Data (Done ONCE) ---
        X_mmap = open_memmap(args.data_path, mode='c')
        n_samples, n_features = X_mmap.shape

        chunk_size = n_samples // world_size
        start = rank * chunk_size
        end = start + chunk_size if rank != world_size - 1 else n_samples

        # Load slice into RAM
        X_local = torch.from_numpy(X_mmap[start:end]).float()

        # --- 3. Pre-allocate Buffers ---
        torch.manual_seed(args.seed)

        W_init = torch.randn(n_features, args.n_components)
        W_init, _ = torch.linalg.qr(W_init, mode='reduced')
        b_init = torch.full((args.n_components,), args.b0)

        G_local = torch.zeros(n_features, args.n_components)
        G_numpy = G_local.numpy()

        # --- 4. Signal Ready ---
        comm.Barrier()
        if rank == 0:
            with open(status_file, "w") as f:
                f.write("READY")

        # --- 5. Event Loop ---
        last_known_cmd = "IDLE"

        while True:
            # Poll control file
            try:
                if os.path.exists(control_file):
                    with open(control_file, "r") as f:
                        data = json.load(f)
                    cmd = data.get("command", "IDLE")
                    n_iter = data.get("n_iter", 0)
                else:
                    cmd = "IDLE"
            except (json.JSONDecodeError, FileNotFoundError, ValueError):
                cmd = last_known_cmd

            # --- STATE MACHINE ---

            # A. EXIT
            if cmd == "EXIT":
                break

            # B. IDLE
            if cmd == "IDLE":
                if last_known_cmd == "RUN":
                    comm.Barrier()
                    if rank == 0:
                        with open(status_file, "w") as f:
                            f.write("READY")
                last_known_cmd = "IDLE"
                time.sleep(0.01)
                continue

            # C. RUN (Trigger Optimization)
            if cmd == "RUN" and last_known_cmd != "RUN":
                last_known_cmd = "RUN"

                W_curr = W_init.clone()
                b_curr = b_init.clone()

                # --- OPTIMIZATION LOOP ---
                for i in range(n_iter):
                    curr_seed = args.seed + i
                    torch.manual_seed(curr_seed)

                    indices = torch.randint(0, len(X_local), (args.batch_size,))
                    X_batch = X_local[indices]

                    torch.matmul(X_batch.T, X_batch @ W_curr, out=G_local)

                    comm.Allreduce(MPI.IN_PLACE, G_numpy, op=MPI.SUM)

                    G_local /= world_size
                    grad_norm_sq = torch.linalg.norm(G_local, dim=0)**2
                    b_curr = torch.sqrt(b_curr**2 + grad_norm_sq)

                    W_curr.addcdiv_(G_local, b_curr.unsqueeze(0), value=1.0)
                    W_curr, _ = torch.linalg.qr(W_curr, mode='reduced')

                # --- FINISH ---
                if rank == 0:
                    # FIX: Use unique temp file to prevent race conditions during rename
                    # if multiple workers incorrectly think they are rank 0.
                    tmp_out_path = f"{out_path}.tmp.{uuid.uuid4().hex}"

                    with open(tmp_out_path, "wb") as f:
                        np.save(f, W_curr.numpy())
                        f.flush()
                        os.fsync(f.fileno()) # Ensure write to disk

                    # Atomic replacement
                    os.rename(tmp_out_path, out_path)

                    with open(status_file, "w") as f:
                        f.write("DONE")

                comm.Barrier()

            time.sleep(0.005)

        if rank == 0:
            print("Worker 0: Received EXIT signal. Shutting down.")


if __name__ == "__main__":
    Solver.entry_point()