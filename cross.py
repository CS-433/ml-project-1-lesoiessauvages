from math_helpers import *
from implementations import *
import numpy as np
import linear_regression as linreg

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
    x_test = x[k_indices[k,:],:]
    y_test = y[k_indices[k,:]]

    k_indices = np.delete(k_indices, k, axis=0).reshape(-1)
    x_train = x[k_indices, :]
    y_train = y[k_indices]

    # print(x.shape)
    # print(y.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # print(x_train.shape)
    # print(y_train.shape)

    # form data with polynomial degree
    phi_tr = build_poly(x_train, degree)
    phi_te = build_poly(x_test, degree)

    # print(phi_tr.shape)
    # print(phi_te.shape)

    # ridge regression
    w, _ = ridge_regression(y_train, phi_tr, lambda_)

    # print(w.shape)

    # calculate the loss for train and test data
    loss_tr = np.sqrt(2*linreg.compute_loss(y_train, phi_tr, w))
    loss_te = np.sqrt(2*linreg.compute_loss(y_test, phi_te, w))

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
            print("Lambda : " + str(lambda_) + " degree : " + str(degree) + " loss : " + str(np.mean(rmse_te3)))
            rmse_te4.append(np.mean(rmse_te3))

        rmse_te.append(np.min(rmse_te4))
        best_lambdas.append(lambdas[np.argmin(rmse_te4)])


    best_rmse = np.min(rmse_te)
    best_lambda = best_lambdas[np.argmin(rmse_te)]
    best_degree = degrees[np.argmin(rmse_te)]

    return best_degree, best_lambda, best_rmse
