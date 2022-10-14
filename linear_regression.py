import numpy as np

def calculate_mse(e):
    return np.mean(e**2)/2

def compute_loss(y, tx, w):
    e = y - tx@w
    return calculate_mse(e)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - tx@w
    grad = -tx.T@e /len(y)
    return grad
