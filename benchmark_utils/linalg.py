import numpy as np


def inv_squared_root(M: np.ndarray):
    U, S, Vh = np.linalg.svd(M)
    return Vh.T @ np.diag(1 / np.sqrt(S)) @ U.T
