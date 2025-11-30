import numpy as np

def worker_oja_step(X_block, W, batch_size, seed):
    """
    Computes a local gradient estimate on a mini-batch of the data block.
    This function is defined here to ensure it is pickleable/hashable by Dask.
    """
    n_local, d = X_block.shape
    rng = np.random.default_rng(seed)

    # Sample mini-batch from the local data block
    indices = rng.integers(0, n_local, batch_size)
    X_b = X_block[indices].T  # (d, batch_size)

    # Oja's gradient approximation
    G = X_b @ (X_b.T @ W) / batch_size
    return G