import os
import sys
import subprocess
import argparse
import inspect
import socket
import pickle
import struct
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np


class DistributedMPISolver(BaseSolver):
    """
    Persistent MPI Solver using Socket Communication.
    Launches workers once in set_objective, then triggers runs via TCP sockets.
    """

    worker_process = None
    server_socket = None
    connection = None

    def set_objective(self, n, d, X_path, n_components):
        self.X_path = X_path
        self.n_components = n_components

        # 1. Setup Socket Server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind to 0.0.0.0 to allow connections from external nodes (if permitted)
        # Port 0 lets the OS choose a free port.
        self.server_socket.bind(("0.0.0.0", 0))
        self.server_socket.listen(1)

        # Get the driver's IP/Hostname and the assigned port
        driver_host = socket.gethostname()
        _, driver_port = self.server_socket.getsockname()

        # 2. Launch Workers (Non-Blocking)
        child_file_path = inspect.getfile(self.__class__)

        # Construct the SLURM/MPI command
        # We pass the driver's host and port to the workers
        cmd = [
            "srun",
            "--overlap",
            "-n", str(self.n_workers),
            "python", child_file_path,
            "--worker",
            "--data_path", self.X_path,
            "--driver_host", str(driver_host),
            "--driver_port", str(driver_port),
            "--n_components", str(self.n_components),
            "--batch_size", str(self.batch_size),
            "--b0", str(self.b0),
            "--seed", str(42)
        ]

        print(f"Driver: Launching persistent MPI cluster on {child_file_path}...")
        print(f"Driver: Listening on {driver_host}:{driver_port}")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

        self.worker_process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env
        )

        # 3. Wait for Connection from Rank 0
        print("Driver: Waiting for worker connection...")
        self.server_socket.settimeout(60)  # Timeout if workers fail to start
        try:
            self.connection, addr = self.server_socket.accept()
            print(f"Driver: Connected to worker at {addr}")
        except socket.timeout:
            self.cleanup()
            raise RuntimeError("Timed out waiting for MPI workers to connect.")

    def run(self, n_iter):
        """
        Trigger the workers to run the optimization loop via socket.
        """
        if not self.connection:
            raise RuntimeError("No active connection to workers.")

        # 1. Send RUN command
        msg = {"command": "RUN", "n_iter": n_iter}
        self._send_msg(self.connection, msg)

        # 2. Receive Result
        # This blocks until Rank 0 sends the result back
        response = self._recv_msg(self.connection)

        if response.get("status") == "DONE":
            self.components = response.get("components")
        else:
            raise RuntimeError(f"Unexpected response from worker: {response}")

    def get_result(self):
        return dict(components=self.components)

    def cleanup(self):
        """
        Terminate workers gracefully.
        """
        if self.connection:
            try:
                self._send_msg(self.connection, {"command": "EXIT"})
                self.connection.close()
            except Exception:
                pass

        if self.server_socket:
            self.server_socket.close()

        if self.worker_process:
            try:
                self.worker_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker_process.kill()

    # --- Socket Helper Methods ---

    @staticmethod
    def _send_msg(sock, msg):
        """Pickles and sends a message with a length header."""
        data = pickle.dumps(msg)
        # Prefix each message with a 4-byte big-endian unsigned integer (network byte order)
        sock.sendall(struct.pack('>I', len(data)) + data)

    @staticmethod
    def _recv_msg(sock):
        """Receives a length header and then the pickled message."""
        # Read message length
        raw_msglen = DistributedMPISolver._recvall(sock, 4)
        if not raw_msglen:
            return None
        msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        data = DistributedMPISolver._recvall(sock, msglen)
        return pickle.loads(data)

    @staticmethod
    def _recvall(sock, n):
        """Helper function to recv n bytes or return None if EOF is hit"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    @classmethod
    def worker(cls, args):
        raise NotImplementedError("Concrete solver must implement 'worker'")

    @classmethod
    def entry_point(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--data_path", type=str)
        # New args for socket connection
        parser.add_argument("--driver_host", type=str)
        parser.add_argument("--driver_port", type=int)

        parser.add_argument("--n_components", type=int)
        parser.add_argument("--batch_size", type=int)
        parser.add_argument("--b0", type=float)
        parser.add_argument("--seed", type=int)

        args, _ = parser.parse_known_args()

        if args.worker:
            cls.worker(args)