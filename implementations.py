import numpy as np



def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


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

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """


    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        gradient= compute_gradient(y, tx, w)
        #update w
        w = w - gamma*gradient
        #compute loss
        loss = compute_loss(y, tx, w)

        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return loss, w


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """


    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute gradient
            gradient= compute_gradient(minibatch_y, minibatch_tx, w)
            #update w
            w = w - gamma*gradient
            #compute loss
            loss = compute_loss(minibatch_y, minibatch_tx, w)

    return loss, w

def least_squares(y, tx):
    """Least squares regression using normal equations.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """

    a = tx.T@tx
    b = tx.T@y
    w = np.linalg.solve(a, b)
    mse = compute_loss(y, tx, w)
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
    loss = compute_loss(y, tx, w)
    return w, loss

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
