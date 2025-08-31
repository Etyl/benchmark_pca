import numpy as np
import scipy.linalg

# Source of Retractions : https://link.springer.com/article/10.1007/s11590-018-1341-z


def uniform(d: int, k: int, random_seed: int) -> np.ndarray:
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


def _tangent_proj(G: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Projection on tangent space of Stiefel Manifold

    Args:
        G (np.ndarray): Matrix (d, k) to project
        W (np.ndarray): Point (d, k) of the Stiefel Manifold for which we consider the tangent space

    Returns:
        np.ndarray: Projection  (d, k) of G on tangent space at W
    """
    wg = W.T @ G
    proj = G - W @ ((wg + wg.T) / 2)
    return proj


def _qr_retraction(W: np.ndarray, G: np.ndarray, step_size: float) -> np.ndarray:
    Q, _ = np.linalg.qr((W - step_size * G), mode="reduced")
    return Q


def _polar_retraction(W: np.ndarray, G: np.ndarray, step_size: float) -> np.ndarray:
    M = W - step_size * G
    U, _, Vh = np.linalg.svd(M, full_matrices=False)
    return U @ Vh


def _A(W: np.ndarray, G: np.ndarray, proj=True) -> np.ndarray:
    if proj:
        pgw = (G - (1 / 2) * W @ (W.T @ G)) @ W.T
    else:
        pgw = G @ W.T
    A = pgw - pgw.T
    return A


def _cayley_retraction(W: np.ndarray, G: np.ndarray, step_size: float, fast=True) -> np.ndarray:
    # source : https://repository.rice.edu/server/api/core/bitstreams/3d52e0a2-b1c6-45ce-84f7-42774b933099/content
    d, k = W.shape
    if fast:
        wg = W.T @ G
        u = np.concatenate([G - (1 / 2) * W @ (W.T @ G), W], axis=1)
        vu = np.zeros((2 * k, 2 * k))
        vu[:k, :k] = wg / 2
        vu[:k, k:] = np.eye(k)
        vu[k:, :k] = -G.T @ G + (wg.T @ wg) / 2
        vu[k:, k:] = -wg / 2
        return W - step_size * u @ np.linalg.inv(np.eye(2 * k) + step_size * vu / 2) @ vu[:, k:]
    else:
        A = _A(W, G)
        return np.linalg.inv(np.eye(d) + step_size * A / 2) @ (np.eye(d) - step_size * A / 2) @ W


def _exp_retraction(W: np.ndarray, G: np.ndarray, step_size: float) -> np.ndarray:
    A = _A(W, G)
    R = scipy.linalg.expm(-step_size * A) @ W
    return R


def rgd_step(W: np.ndarray, G: np.ndarray, step_size: float, mode="QR") -> np.ndarray:
    """Perform one step of RGD on Stiefel Manifold

    Args:
        W (np.ndarray): Current point on the Stiefel manifold
        G (np.ndarray): Non-projected euclidean gradient of objectif to minimize at W
        mode (str, optional): Type of retraction, one of ["QR", "polar", "cayley", "exp"]. Defaults to "QR".

    Returns:
        np.ndarray: Point of the Stiefel manifold, obtained by following G direction from W point
    """
    G_proj = _tangent_proj(G, W)

    if mode == "QR":
        return _qr_retraction(W, G_proj, step_size)
    if mode == "polar":
        return _polar_retraction(W, G_proj, step_size)
    if mode == "cayley":
        return _cayley_retraction(W, G_proj, step_size)
    if mode == "exp":
        return _exp_retraction(W, G_proj, step_size)

    raise ValueError("'mode' should be one of ['QR', 'polar', 'cayley', 'exp']")
