def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """


def least_squares(y, tx):
    """Least squares regression using normal equations.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """

    w = np.linalg.solve(tx.T@tx, tx.T@y)
    w = w.squeeze()
    mse = np.mean((y-tx@w)**2)
    return (w, mse)

def ridge_regression(y, tx, lambda ):
    """Ridge regression using normal equations.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """
    l = lambda_*2*y.size
    return np.linalg.solve(tx.T@tx + l*np.eye(tx.shape[1]), tx.T@y)

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1}).
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """


def reg_logistic_regression(y, tx, lambda, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent
    or SGD (y ∈ {0, 1}, with regularization term λ∥w∥2 )
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """
