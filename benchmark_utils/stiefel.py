import numpy as np
import scipy.linalg

# Source of Retractions : https://link.springer.com/article/10.1007/s11590-018-1341-z


def tangent_proj(W: np.ndarray, X: np.ndarray) -> np.ndarray :
    """Projection on tangent space of Stiefel Manifold

    Args:
        W (np.ndarray): Matrix (d, k) to project
        X (np.ndarray): Point (d, k) of the Stiefel Manifold for which we consider the tangent space

    Returns:
        np.ndarray: Projection  (d, k) of W on tangent space at X
    """
    xw = X.T@W 
    proj = W - X@((xw + xw.T)/2) 
    return proj 


def _qr_retraction(X: np.ndarray, V: np.ndarray) -> np.ndarray: 
    Q, _ = np.linalg.qr((X+V), mode="reduced")
    return Q

def _polar_retraction(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    eig, U = np.linalg.eigh(np.eye(X.shape[1]) + V.T @ V)
    squared_root_inv = U @ np.diag(1/np.sqrt(eig)) @ U.T
    return (X + V) @ squared_root_inv

def _A(X: np.ndarray, V:np.ndarray) -> np.ndarray:
    pvx = (np.eye(X.shape[0]) - (X @ X.T)/2) @ V @ X.T 
    A = pvx - pvx.T 
    return A

def _cayley_retraction(X: np.ndarray, V: np.ndarray) -> np.ndarray: 
    #TODO: Implement efficient implementation based on the fact that rank A <= 2k (SMW Formula)
    # source : https://repository.rice.edu/server/api/core/bitstreams/3d52e0a2-b1c6-45ce-84f7-42774b933099/content
    A = _A(X, V)
    R = np.linalg.inv(np.eye(X.shape[0]) - A/2) @ (np.eye(X.shape[0]) + A/2) @ X
    return R

def _exp_retraction(X: np.ndarray, V: np.ndarray) -> np.ndarray: 
    A = _A(X, V)
    R = scipy.linalg.expm(A) @ X
    return R

def retraction(X: np.ndarray, V: np.ndarray, mode="QR") -> np.ndarray:
    """Approximate exponential map on the Stiefel manifold

    Args:
        X (np.ndarray): Departure point on the Stiefel manifold
        V (np.ndarray): Vector in the tangent space of X 
        mode (str, optional): Type of retraction, one of ["QR", "polar", "cayley", "exp"]. Defaults to "QR".

    Returns:
        np.ndarray: Point of the Stiefel manifold, obtained by following V direction from X point
    """
    if mode == "QR":
        return _qr_retraction(X, V)
    if mode == "polar":
        return _polar_retraction(X, V)
    if mode == "cayley" :
        return _cayley_retraction(X, V)
    if mode == "exp" : 
        return _exp_retraction(X, V)
    raise ValueError("'mode' should be one of ['QR', 'polar', 'cayley', 'exp']")