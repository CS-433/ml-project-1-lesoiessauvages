def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    returns :
        w : Last weight vector
        loss : coresponding loss"""

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent
    returns :
        w : Last weight vector
        loss : coresponding loss"""

def least_squares(y, tx):
    """Least squares regression using normal equations
    returns :
        w : Last weight vector
        loss : coresponding loss"""

def ridge_regression(y, tx, lambda ):
    """Ridge regression using normal equations
    returns :
        w : Last weight vector
        loss : coresponding loss"""

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1})
    returns :
        w : Last weight vector
        loss : coresponding loss"""

def reg_logistic_regression(y, tx, lambda, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent
    or SGD (y ∈ {0, 1}, with regularization term λ∥w∥2 )
    returns :
        w : Last weight vector
        loss : coresponding loss"""
