import numpy as np


def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return 1.0 / (1 + np.exp(-t))


def compute_loss(y, tx, w, use_epsilon=False):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        use_epsilon: if set to True, uses a formula with extra variable epsilon to avoid invalid logarithms

    Returns:
        a non-negative loss

    """

    # print("begin compute loss")
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx @ w)

    if use_epsilon:
        ### VERSION WITH EPSILON TO AVOID INVALID VALUES
        epsilon = 1e-7
        loss = y.T @ np.log(pred + epsilon) + (1 - y).T @ np.log(1 - pred + epsilon)
    else:
        ### CLASSICAL VERSION OF THE FORMULA
        loss = y.T @ np.log(pred) + (1 - y).T @ np.log(1 - pred)

    return (-loss.item()) / y.shape[0]


def compute_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    pred = sigmoid(tx @ w)

    grad = tx.T @ (pred - y) / y.shape[0]

    return grad


def penalized_logistic_regression_gradient(y, tx, w, lambda_):

    gradient = compute_gradient(y, tx, w) + 2 * lambda_ * w
    return gradient
