import numpy as np
import linear_regression as linreg

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


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    output = np.ones(x.shape[0])
    for i in range(1, degree+1):
        output = np.c_[output, x**i]
    return output

def standardize(x):
    """Standardize the original data set column-wise."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def roundToPrediction(a):
    a[a < 0] = -1
    a[a >= 0] = 1
    return a

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527598144, 0.3355591436129497)
    """

    # get k'th subgroup in test, others in train
    x_test = x[k_indices[k,:]]
    y_test = y[k_indices[k,:]]

    k_indices = np.delete(k_indices, k, axis=0).reshape(-1)
    x_train = x[k_indices]
    y_train = y[k_indices]

    # form data with polynomial degree
    x_tr = build_poly(x_train, degree)
    x_te = build_poly(x_test, degree)

    # ridge regression
    w = ridge_regression(y_train, x_tr, lambda_)

    # calculate the loss for train and test data
    loss_tr = np.sqrt(2*linreg.compute_mse(y_train, x_tr, w))
    loss_te = np.sqrt(2*linreg.compute_mse(y_test, x_te, w))

    return loss_tr, loss_te



def best_degree_selection(y, x, degrees, k_fold, lambdas, seed = 1):
    """cross validation over regularisation parameter lambda and degree.

    Args:
        degrees: shape = (d,), where d is the number of degrees to test
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)

    >>> best_degree_selection(np.arange(2,11), 4, np.logspace(-4, 0, 30))
    (7, 0.004520353656360241, 0.2895728056812453)
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)


    # cross validation over degrees and lambdas
    # TODO: Rename variables
    rmse_te = []
    best_lambdas = []


    for degree in degrees:
        rmse_te4 = []
        for lambda_ in lambdas:
            rmse_te3 = []
            for k in range(k_fold):
                rmse_tr2, rmse_te2 = cross_validation(y, x, k_indices, k, lambda_, degree)
                rmse_te3.append(rmse_te2)
            rmse_te4.append(np.mean(rmse_te3))

        rmse_te.append(np.min(rmse_te4))
        best_lambdas.append(lambdas[np.argmin(rmse_te4)])


    best_rmse = np.min(rmse_te)
    best_lambda = best_lambdas[np.argmin(rmse_te)]
    best_degree = degrees[np.argmin(rmse_te)]

    return best_degree, best_lambda, best_rmse
