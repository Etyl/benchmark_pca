import numpy as np


# Source of Retractions 


def tangent_proj(W: np.ndarray, X: np.ndarray) -> np.ndarray :
    """Projection on tangent space of Stiefel Manifold

    Args:
        W (np.ndarray): Matrix to project
        X (np.ndarray): Point of the Stiefel Manifold for which we consider the tangent space

    Returns:
        np.ndarray: Projection of W on tangent space at X
    """
    xw = X@W.T 
    proj = W - ((xw + xw.T)/2) @ X.T 
    return proj 

def _qr_retraction(X: np.ndarray, V: np.ndarray) -> np.ndarray: 
    Q, _ = np.linalg.qr((X+V).T, mode="reduced")
    return Q.T 

def _polar_retraction(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    eig, U = np.linalg.eigh(np.eye(X.shape[0]) + V @ V.T)
    squared_root_inv = U @ np.diag(1/np.sqrt(eig)) @ U.T 
    return squared_root_inv @ (X + V)

def _cayley_retraction(X: np.ndarray, V: np.ndarray) -> np.ndarray: 
    # For efficient cayley retraction : https://repository.rice.edu/server/api/core/bitstreams/3d52e0a2-b1c6-45ce-84f7-42774b933099/content
    return NotImplementedError("Efficient Cayley retraction not implemented and naive too expensive")

def _exponential_retr(X: np.ndarray, V: np.ndarray) -> np.ndarray: 
    Q, R = np.linalg.qr(-V.T@(np.eye(X.shape[0]) - X.T @ X))