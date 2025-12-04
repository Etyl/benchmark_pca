import numpy as np


def uniform(d: int, k: int, random_seed: int = None) -> np.ndarray:
    """Uniform sampling on the Stiefel Manifold

    Args:
        d (int): dimension of each vector in the orthogonal family
        k (int): number of orthogonal vectors

    Returns:
        np.ndarray: Uniformly sampled point on the Stiefel manifold
    """
    generator = np.random.default_rng(random_seed)

    W = generator.normal(0, 1, (d, k))
    W, _ = np.linalg.qr(W, mode="reduced")
    return W
