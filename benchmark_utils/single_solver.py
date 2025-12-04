import os
import sys
import uuid
import inspect
import argparse
import subprocess
import socket
import pickle
import struct
import atexit
import numpy as np
from benchopt import BaseSolver


class SingleNodeSolver(BaseSolver):
    """
    Abstract Base Class for Persistent Single-Node Solvers via SLURM.
    Launches a worker in warm_up (via srun) and triggers runs via TCP socket.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Internal state
        self.worker_process = None
        self.server_socket = None
        self.connection = None
        # Ensure workers are killed if the script exits abruptly
        atexit.register(self.cleanup)

    def __del__(self):
        # Ensure cleanup is called when the solver object is destroyed
        self.cleanup()

    def set_objective(self, n, d, X_path, n_components):
        # Store parameters; deferred launch to warm_up
        self.n = n
        self.d = d
        self.X_path = X_path
        self.n_components = n_components

    def warm_up(self):
        """
        Launch the worker and run one iteration to warm up the system.
        """
        # Ensure any previous workers are cleaned up
        self.cleanup()

        # 1. Setup Socket Server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(("0.0.0.0", 0))
        self.server_socket.listen(1)

        driver_host = socket.gethostname()
        _, driver_port = self.server_socket.getsockname()

        # 2. Identify script and build srun command
        script_path = inspect.getfile(self.__class__)

        # Unique ID for this solver instance
        worker_id = uuid.uuid4().hex[:8]

        cmd = [
            "srun",
            "python", script_path,
            "--worker",
            "--X_path", self.X_path,
            "--n_components", str(self.n_components),
            # Pass connection details
            "--driver_host", driver_host,
            "--driver_port", str(driver_port)
        ]

        # Append solver-specific parameters
        for param_name in self.parameters:
            param_value = getattr(self, param_name)
            cmd.extend([f"--{param_name}", str(param_value)])

        print(f"[{self.name}] Launching worker {worker_id} on {driver_host}:{driver_port}...")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

        # 3. Launch Worker (Non-Blocking)
        self.worker_process = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env
        )

        # 4. Wait for Connection
        self.server_socket.settimeout(120)
        try:
            self.connection, addr = self.server_socket.accept()
            print(f"[{self.name}] Worker connected from {addr}")
        except socket.timeout:
            raise RuntimeError("Timed out waiting for SLURM worker to connect.")

    def run(self, n_iter):
        # If no connection (e.g. wiped by previous get_result or no warm_up), launch now.
        if not self.connection:
            raise RuntimeError("No active connection to workers. Please call warm_up() first.")

        # 1. Send RUN command
        msg = {"command": "RUN", "n_iter": n_iter}
        self._send_msg(self.connection, msg)

        # 2. Wait for Result
        response = self._recv_msg(self.connection)

        if response and response.get("status") == "DONE":
            self.components = response.get("components")
        else:
            raise RuntimeError(f"Worker failed or sent invalid response: {response}")

    def get_result(self):
        # Capture result
        res = dict(components=self.components)

        return res

    def cleanup(self):
        """Terminate worker and close sockets."""
        # 1. Send EXIT command to workers
        if getattr(self, 'connection', None):
            try:
                self._send_msg(self.connection, {"command": "EXIT"})
                self.connection.close()
            except Exception:
                pass
            self.connection = None

        # 2. Close Server Socket
        if getattr(self, 'server_socket', None):
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        # 3. Kill Process
        if getattr(self, 'worker_process', None):
            try:
                self.worker_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.worker_process.kill()
            self.worker_process = None

    # --- Worker Logic ---

    @classmethod
    def worker(cls, args):
        """
        Worker Entry Point: Loads data once, then loops waiting for commands.
        """
        # 1. Load Data
        if not os.path.exists(args.X_path):
            print(f"Error: Data file {args.X_path} not found.")
            sys.exit(1)

        X = np.load(args.X_path)

        # 2. Connect to Driver
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((args.driver_host, args.driver_port))
        except Exception as e:
            print(f"Worker failed to connect to driver: {e}")
            sys.exit(1)

        # 3. Event Loop
        while True:
            msg = cls._recv_msg(sock)
            if not msg:
                break

            cmd = msg.get("command")

            if cmd == "EXIT":
                break

            if cmd == "RUN":
                # Update iteration count in args for the solve method
                args.n_iter = msg.get("n_iter")

                try:
                    # Run the concrete solver logic
                    components = cls.solve(X, args)

                    # Send back results
                    response = {"status": "DONE", "components": components}
                    cls._send_msg(sock, response)
                except Exception as e:
                    print(f"Worker error during solve: {e}")
                    # Optionally send error back
                    break

        sock.close()

    @classmethod
    def solve(cls, X, args):
        """
        Abstract method: Must be implemented by the concrete solver.
        Returns the computed components (W).
        """
        raise NotImplementedError("Subclasses must implement 'solve'")

    @classmethod
    def entry_point(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker", action="store_true")
        parser.add_argument("--X_path", type=str)
        parser.add_argument("--driver_host", type=str)
        parser.add_argument("--driver_port", type=int)
        parser.add_argument("--n_components", type=int)
        # n_iter is optional here as it comes via socket for RUN,
        # but kept for compatibility or initial parsing
        parser.add_argument("--n_iter", type=int, default=0)

        # Add solver-specific parameters dynamically
        for param, values in cls.parameters.items():
            param_type = type(values[0]) if values else str
            parser.add_argument(f"--{param}", type=param_type)

        args, _ = parser.parse_known_args()

        if args.worker:
            cls.worker(args)

    # --- Socket Helpers ---
    @staticmethod
    def _send_msg(sock, msg):
        try:
            data = pickle.dumps(msg)
            sock.sendall(struct.pack('>I', len(data)) + data)
        except (OSError, BrokenPipeError):
            pass

    @staticmethod
    def _recv_msg(sock):
        try:
            raw_msglen = SingleNodeSolver._recvall(sock, 4)
            if not raw_msglen:
                return None
            msglen = struct.unpack('>I', raw_msglen)[0]
            data = SingleNodeSolver._recvall(sock, msglen)
            return pickle.loads(data)
        except (OSError, struct.error):
            return None

    @staticmethod
    def _recvall(sock, n):
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except OSError:
                return None
        return data