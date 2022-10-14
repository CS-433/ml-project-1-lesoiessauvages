from math_helpers import *
import numpy as np
import logistic_regression
import linear_regression


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """

    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        gradient= linear_regression.compute_gradient(y, tx, w)
        #update w
        w = w - gamma*gradient
        #compute loss
        loss = linear_regression.compute_loss(y, tx, w)

        ##print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              ##bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """

    np.random.seed(1)

    w = initial_w
    batch_size = 100
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute gradient
            gradient = linear_regression.compute_gradient(minibatch_y, minibatch_tx, w)
            #update w
            w = w - gamma*gradient
            #compute loss
            loss = linear_regression.compute_loss(minibatch_y, minibatch_tx, w)
            print(loss)
            ##if(loss < 0.001):
                ##return w, loss


    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """

    a = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(a, b)
    mse = linear_regression.compute_loss(y, tx, w)
    return w, mse


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """
    lI = lambda_*2*len(y)*np.eye(tx.shape[1])
    a = tx.T@tx + lI
    b = tx.T@y
    w = np.linalg.solve(a, b)
    loss = linear_regression.compute_loss(y, tx, w)

    print(loss)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1}).
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """
    w = initial_w

    for iter in range(max_iter):
        grad = logistic_regression.compute_gradient(y, tx, w)
        w -= gamma * grad
        loss = logistic_regression.compute_loss(y, tx, w)

    return w, loss



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent
    or SGD (y ∈ {0, 1}, with regularization term λ∥w∥2 )
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """
