import os
import sys
import time
import subprocess
import argparse
import tempfile
import inspect
import json
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class DistributedMPISolver(BaseSolver):
    """
    Persistent MPI Solver.
    Launches workers once in set_objective, then triggers runs via shared files.
    """

    tmp_dir = None
    worker_process = None

    def set_objective(self, n, d, X_path, n_components):
        self.X_path = X_path
        self.n_components = n_components

        # 1. Setup Shared Control Directory
        # We use a temp directory that is shared between the driver and workers
        self.tmp_dir = tempfile.mkdtemp(prefix="benchopt_mpi_")
        self.control_file = os.path.join(self.tmp_dir, "control.json")
        self.status_file = os.path.join(self.tmp_dir, "status_rank_0.txt")
        self.out_path = os.path.join(self.tmp_dir, "results.npy")

        # Initialize control file with IDLE state
        self._write_control("IDLE", 0)

        # 2. Launch Workers (Non-Blocking)
        child_file_path = inspect.getfile(self.__class__)

        # Construct the SLURM command
        cmd = [
            "srun",
            "--overlap",
            "-n", str(self.n_workers),
            "python", child_file_path,
            "--worker",
            "--data_path", self.X_path,
            "--tmp_dir", self.tmp_dir, # Pass temp dir for coordination
            "--n_components", str(self.n_components),
            "--batch_size", str(self.batch_size),
            "--b0", str(self.b0),
            "--seed", str(42)
        ]

        print(f"Driver: Launching persistent MPI cluster on {child_file_path}...")

        # Prepare environment: Workers need to find 'benchmark_utils'
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

        # Launch process in background (Popen instead of run)
        self.worker_process = subprocess.Popen(
            cmd,
            stdout=sys.stdout, # Stream worker logs to console
            stderr=sys.stderr,
            env=env
        )

        # 3. Wait for Workers to be Ready
        print("Driver: Waiting for workers to initialize...")
        try:
            self._wait_for_status("READY")
        except RuntimeError as e:
            # If workers fail immediately (e.g., import error), clean up and raise
            self.cleanup()
            raise e

        print("Driver: Workers are ready.")

    def run(self, n_iter):
        """
        Trigger the workers to run the optimization loop.
        """
        if not self.worker_process or self.worker_process.poll() is not None:
            raise RuntimeError("MPI Worker process is not running.")

        # 1. Send RUN command with the number of iterations
        self._write_control("RUN", n_iter)

        # 2. Wait for completion signal from Rank 0
        self._wait_for_status("DONE")

        # 3. Load Result
        if os.path.exists(self.out_path):
            self.components = np.load(self.out_path)
        else:
            raise FileNotFoundError("Workers finished but no output file found.")

        # 4. Reset workers to IDLE for next run
        # This handshake ensures workers are ready to accept a new command
        self._write_control("IDLE", 0)
        self._wait_for_status("READY")

    def get_result(self):
        return dict(components=self.components)

    def cleanup(self):
        """
        Terminate workers gracefully.
        """
        # Send EXIT command
        if self.tmp_dir and os.path.exists(self.tmp_dir):
            try:
                self._write_control("EXIT", 0)
            except Exception:
                pass

        # Wait for process to exit
        if self.worker_process:
            try:
                self.worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker_process.kill()

        import shutil
        if self.tmp_dir and os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)

    # --- Helper Methods ---
    def _write_control(self, command, n_iter):
        """Writes a command to the shared control file atomically."""
        data = {"command": command, "n_iter": n_iter}
        tmp_name = self.control_file + ".tmp"
        with open(tmp_name, "w") as f:
            json.dump(data, f)
        # Atomic rename prevents workers from reading partial JSON
        os.rename(tmp_name, self.control_file)

    def _wait_for_status(self, target_status):
        """Polls the status file until it matches target."""
        while True:
            # Check if status file exists and contains target
            if os.path.exists(self.status_file):
                try:
                    with open(self.status_file, "r") as f:
                        content = f.read().strip()
                    if content == target_status:
                        return
                except (IOError, ValueError):
                    pass # Retry on read error

            # Check if subprocess died unexpectedly
            if self.worker_process.poll() is not None:
                raise RuntimeError("MPI Workers died unexpectedly during wait.")

            time.sleep(0.01)

    @classmethod
    def worker(cls, args):
        raise NotImplementedError("Concrete solver must implement 'worker'")

    @classmethod
    def entry_point(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--tmp_dir", type=str)
        parser.add_argument("--n_components", type=int)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--b0", type=float)
        parser.add_argument("--seed", type=int)

        args, _ = parser.parse_known_args()

        if args.worker:
            cls.worker(args)