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

    #print("begin compute loss")
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    epsilon = 1e-7
    #epsilon = 0

    # print("w : " + str(w))
    # print("tx@w : " + str(tx@w))
    pred = sigmoid(tx@w)
    #print("pred(min,max) : " + str(np.min(pred)) + " " + str(np.max(pred)))

    loss = y.T@np.log(pred + epsilon) + (1 - y).T@np.log(1 - pred + epsilon)

    #T - Log(e^t + 1)

    #return np.squeeze(- loss).item()#1/2*??
    return 1/2*(-loss.item())/y.shape[0]


def compute_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """

    # print(w.shape)
    pred = sigmoid(tx@w)
    # print((pred-y).shape)
    grad = np.mean((pred - y)*tx.T, 1)
    # print(grad.shape)
    return grad


def penalized_logistic_regression_gradient(y, tx, w, lambda_):

    gradient = compute_gradient(y, tx, w) + 2 * lambda_ * w
    return gradient

def penalized_logistic_regression_loss(y, tx, w, lambda_):

    loss = compute_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    return loss
