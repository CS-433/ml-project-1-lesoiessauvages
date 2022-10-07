def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """
    def compute_loss(y, tx, w):
        e = y - tx@w
        return np.mean(e**2)/2

    def compute_gradient(y, tx, w):
        """Compute the gradient."""
        e = y - tx@w
        return -1/y.shape[0] * tx.T@e

    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - gamma*gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """

    def compute_stoch_gradient(y, tx, w):
        """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
        e = y - tx@w
        grad = -tx.T@e / e.size
        return grad

    losses = []
    ws = [initial_w]
    w = initial_w
    for n_iter in range(max_iters):
        minibatch_y, minibatch_tx = batch_iter(y, tx, batch_size).__next__()
        loss = compute_loss(minibatch_y, minibatch_tx, w)
        losses.append(loss)
        gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
        w = w - gamma*gradient
        ws.append(w)


    return losses, ws

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
