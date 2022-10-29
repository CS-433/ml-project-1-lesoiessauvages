import numpy as np


def compute_loss(y, tx, w):
    """Computes MSE"""
    e = y - tx @ w
    return np.mean(e ** 2) / 2


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx @ w
    grad = -tx.T @ e / y.shape[0]
    return grad
