import os
import sys
import socket
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

    @classmethod
    def worker(cls, args):
        # --- 1. Init Environment & MPI ---
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK", None)
        if slurm_cpus:
            torch.set_num_threads(int(slurm_cpus))
        else:
            torch.set_num_threads(4)

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        # Handle SLURM rank detection overlap issues
        if "SLURM_PROCID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
        if "SLURM_NTASKS" in os.environ:
            world_size = int(os.environ["SLURM_NTASKS"])

        # Debug print
        print(f"Worker initialized: Rank {rank}/{world_size} on {socket.gethostname()}")
        sys.stdout.flush()

        # --- 2. Load Data ---
        X_mmap = open_memmap(args.data_path, mode='c')
        n_samples, n_features = X_mmap.shape
        chunk_size = n_samples // world_size
        start = rank * chunk_size
        end = start + chunk_size if rank != world_size - 1 else n_samples
        X_local = torch.from_numpy(X_mmap[start:end]).float()

        # --- 3. Establish Connection (Rank 0 only) ---
        sock = None
        if rank == 0:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((args.driver_host, args.driver_port))
                print("Worker 0: Connected to Driver.")
            except Exception as e:
                print(f"Worker 0: Connection failed: {e}")
                sys.exit(1)

        # --- 4. Event Loop ---
        while True:
            cmd_data = None

            # Rank 0 waits for command from Driver
            if rank == 0:
                try:
                    cmd_data = DistributedMPISolver._recv_msg(sock)
                except Exception:
                    cmd_data = {"command": "EXIT"} # Fallback on error

            # Broadcast command to all workers
            cmd_data = comm.bcast(cmd_data, root=0)

            if cmd_data is None:
                break

            cmd = cmd_data.get("command")

            # --- A. EXIT ---
            if cmd == "EXIT":
                break

            # --- B. RUN ---
            if cmd == "RUN":
                n_iter = cmd_data.get("n_iter", 0)

                # Setup randomness
                torch.manual_seed(args.seed)

                # Re-init weights for every run (Stateful caching could be added here)
                # Note: To match previous behavior, we might want to cache W between runs
                # but currently the driver asks for fresh runs usually.
                # Assuming fresh start or simple continuation based on benchopt flow.
                # Here we re-init to be safe as per previous code logic.

                W_curr = torch.randn(n_features, args.n_components)
                W_curr, _ = torch.linalg.qr(W_curr, mode='reduced')
                b_curr = torch.full((args.n_components,), args.b0)

                G_local = torch.zeros(n_features, args.n_components)
                G_numpy = G_local.numpy()

                # Optimization Loop
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

                # Send results back (Rank 0 only)
                if rank == 0:
                    result = {
                        "status": "DONE",
                        "components": W_curr.numpy()
                    }
                    DistributedMPISolver._send_msg(sock, result)

            # Wait for all ranks before next command loop
            comm.Barrier()

        if rank == 0 and sock:
            sock.close()
            print("Worker 0: Socket closed.")

if __name__ == "__main__":
    Solver.entry_point()