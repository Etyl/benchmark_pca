import numpy as np
import scipy.linalg

# Source of Retractions : https://link.springer.com/article/10.1007/s11590-018-1341-z


def tangent_proj(G: np.ndarray, W: np.ndarray) -> np.ndarray :
    """Projection on tangent space of Stiefel Manifold

    Args:
        G (np.ndarray): Matrix (d, k) to project
        W (np.ndarray): Point (d, k) of the Stiefel Manifold for which we consider the tangent space

    Returns:
        np.ndarray: Projection  (d, k) of G on tangent space at W
    """
    wg = W.T@G
    proj = G - W@((wg + wg.T)/2) 
    return proj 


def _qr_retraction(W: np.ndarray, G: np.ndarray) -> np.ndarray: 
    Q, _ = np.linalg.qr((W+G), mode="reduced")
    return Q

def _polar_retraction(W: np.ndarray, G: np.ndarray) -> np.ndarray:
    eig, U = np.linalg.eigh(np.eye(W.shape[1]) + G.T @ G)
    squared_root_inv = U @ np.diag(1/np.sqrt(eig)) @ U.T
    return (W + G) @ squared_root_inv

def _A(W: np.ndarray, G:np.ndarray) -> np.ndarray:
    pgw = (np.eye(W.shape[0]) - (W @ W.T)/2) @ G @ W.T 
    A = pgw - pgw.T 
    return A

def _cayley_retraction(W: np.ndarray, G: np.ndarray) -> np.ndarray: 
    #TODO: Implement efficient implementation based on the fact that rank A <= 2k (SMW Formula)
    # source : https://repository.rice.edu/server/api/core/bitstreams/3d52e0a2-b1c6-45ce-84f7-42774b933099/content
    A = _A(W, G)
    R = np.linalg.inv(np.eye(W.shape[0]) - A/2) @ (np.eye(W.shape[0]) + A/2) @ W
    return R

def _exp_retraction(W: np.ndarray, G: np.ndarray) -> np.ndarray: 
    A = _A(W, G)
    R = scipy.linalg.expm(A) @ W
    return R

def retraction(W: np.ndarray, G: np.ndarray, mode="QR") -> np.ndarray:
    """Approximate exponential map on the Stiefel manifold

    Args:
        W (np.ndarray): Departure point on the Stiefel manifold
        G (np.ndarray): Vector in the tangent space of W
        mode (str, optional): Type of retraction, one of ["QR", "polar", "cayley", "exp"]. Defaults to "QR".

    Returns:
        np.ndarray: Point of the Stiefel manifold, obtained by following G direction from W point
    """
    if mode == "QR":
        return _qr_retraction(W, G)
    if mode == "polar":
        return _polar_retraction(W, G)
    if mode == "cayley" :
        return _cayley_retraction(W, G)
    if mode == "exp" : 
        return _exp_retraction(W, G)
    raise ValueError("'mode' should be one of ['QR', 'polar', 'cayley', 'exp']")