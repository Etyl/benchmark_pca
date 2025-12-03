import numpy as np


def worker_oja_step(X_block, W, batch_size, seed):
    n_local, d = X_block.shape
    rng = np.random.default_rng(seed)

    # Sample mini-batch from the local data block
    indices = rng.integers(0, n_local, batch_size)
    X_b = X_block[indices].T  # (d, batch_size)

    # Oja's gradient approximation
    G = X_b @ (X_b.T @ W) / batch_size
    return G


def worker_weight_step(
    X_block, W, step_size, batch_size, seed, proj_after_update
):
    """
    Computes a local update step: Gradient -> Update -> Optional Projection.
    Returns the updated local weight matrix W_local.
    """
    n_local, d = X_block.shape
    rng = np.random.default_rng(seed)

    # Sample mini-batch
    indices = rng.integers(0, n_local, batch_size)
    X_b = X_block[indices].T  # (d, batch_size)

    # Calculate Gradient (Oja)
    G = X_b @ (X_b.T @ W) / batch_size

    # Local Update
    W_local = W + step_size * G

    # Optional Local Projection
    if proj_after_update:
        W_local, _ = np.linalg.qr(W_local, mode="reduced")

    return W_local
