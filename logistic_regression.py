import numpy as np

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array

    """
    return 1.0 / (1 + np.exp(-t))


def compute_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss).item()


def compute_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)

    array([[-0.20741526],
           [ 0.4134208 ],
           [ 1.03425686]])
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def penalized_logistic_regression_gradient(y, tx, w, lambda_):

    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return gradient

def penalized_logistic_regression_loss(y, tx, w, lambda_):

    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return loss
