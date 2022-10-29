from math_utils import *
from implementations import *
import numpy as np
import linear_regression as linreg

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_lambda_degree(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        test root mean square errors rmse = sqrt(2 mse)
    """

    # get k'th subgroup in test, others in train
    x_test = x[k_indices[k,:],:]
    y_test = y[k_indices[k,:]]

    k_indices = np.delete(k_indices, k, axis=0).reshape(-1)
    x_train = x[k_indices, :]
    y_train = y[k_indices]


    # form data with polynomial degree
    phi_tr = build_poly(x_train, degree)
    phi_te = build_poly(x_test, degree)

    # ridge regression
    w, _ = ridge_regression(y_train, phi_tr, lambda_)

    # calculate the loss for train and test data
    loss_te = np.sqrt(2*linreg.compute_loss(y_test, phi_te, w))

    return loss_te



def cross_validation_lambda_degree_gamma(y, x, k_indices, k, lambda_, degree, gamma):
    """return the loss of penalized_logistic_regression regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()
        gamma:      scalar, cf. gradient descent

    Returns:
        test root mean square errors rmse = sqrt(2 mse)

    """

    # get k'th subgroup in test, others in train
    x_test = x[k_indices[k,:],:]
    y_test = y[k_indices[k,:]]

    k_indices = np.delete(k_indices, k, axis=0).reshape(-1)
    x_train = x[k_indices, :]
    y_train = y[k_indices]


    # form data with polynomial degree
    phi_tr = build_poly(x_train, degree)
    phi_te = build_poly(x_test, degree)

    max_iters = 100
    initial_w = np.ones(phi_tr.shape[1])

    # ridge regression
    w, _ = mean_squared_error_gd(y_train, phi_tr, initial_w, max_iters, gamma)

    # calculate the loss for train and test data
    loss_te = np.sqrt(2*logreg.compute_loss(y_test, phi_te, w))

    return loss_te



def best_degree_lambda_selection(y, x, degrees, k_fold, lambdas, seed = 1):
    """cross validation over regularisation parameter lambda and degree.

    Args:
        degrees: shape = (d,), where d is the number of degrees to test
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)

    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)


    # cross validation over degrees and lambdas
    rmse_te = []
    best_lambdas = []


    for degree in degrees:
        rmse_te_for_degree = []
        for lambda_ in lambdas:
            sum_rmse_te = 0
            for k in range(k_fold):
                sum_rmse_te += cross_validation_lambda_degree(y, x, k_indices, k, lambda_, degree)

            print("Lambda : " + str(lambda_) + " degree : " + str(degree) + " loss : " + str(sum_rmse_te/k_fold))
            rmse_te_for_degree.append(sum_rmse_te/k_fold)

        rmse_te.append(np.min(rmse_te_for_degree))
        best_lambdas.append(lambdas[np.argmin(rmse_te_for_degree)])


    best_rmse = np.min(rmse_te)
    best_lambda = best_lambdas[np.argmin(rmse_te)]
    best_degree = degrees[np.argmin(rmse_te)]

    print("Best lambda : " + str(best_lambda) + " best degree : " + str(best_degree) + " loss : " + str(best_rmse))

    return best_degree, best_lambda, best_rmse



def best_degree_lambda_gamma_selection(y, x, degrees, k_fold, lambdas, gammas, seed = 1):
    """cross validation over regularisation parameter lambda and degree and learning rate gamma.

    Args:
        degrees: shape = (d,), where d is the number of degrees to test
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
        gammas: shape = (g, ) where g is the number of values of gamma to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_gamma : scalar, value of the best gamma
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)

    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)


    # cross validation over degrees and lambdas

    best_lambda_for_degree = []
    best_gamma_for_degree = []
    best_rmse_for_degree = []
    for degree in degrees:
        best_gamma_for_lambda = []
        best_rmse_for_lambda = []
        for lambda_ in lambdas:
            rmse_for_degree_lambda_gamma = []
            for gamma in gammas:
                sum_rmse_te = 0
                for k in range(k_fold):
                    sum_rmse_te += cross_validation_lambda_degree_gamma(y, x, k_indices, k, lambda_, degree, gamma)
                print("lambda : " + str(lambda_) + " degree : " + str(degree) + " gamma : " + str(gamma) + " loss : " + str(sum_rmse_te/k_fold))
                rmse_for_degree_lambda_gamma.append(sum_rmse_te/k_fold)
            best_rmse_for_lambda.append(np.min(rmse_for_degree_lambda_gamma))
            best_gamma_for_lambda.append(gammas[np.argmin(rmse_for_degree_lambda_gamma)])
        best_rmse_for_degree.append(np.min(best_rmse_for_lambda))
        best_lambda_for_degree.append(lambdas[np.argmin(best_rmse_for_lambda)])
        best_gamma_for_degree.append(gammas[np.argmin(best_rmse_for_lambda)])
    best_rmse = np.min(best_rmse_for_degree)
    best_lambda = lambdas[np.argmin(best_rmse_for_degree)]
    best_gamma = gammas[np.argmin(best_rmse_for_degree)]
    best_degree = degrees[np.argmin(best_rmse_for_degree)]

    print("Best lambda : " + str(best_lambda) + " best degree : " + str(best_degree) + " best gamma : " + str(best_gamma) +  " loss : " + str(best_rmse))

    return best_degree, best_lambda, best_gamma, best_rmse



def split_data(y, x, k_fold, seed=1):
    """
    Simple function to split data. Returns a training set and associated labels,
    as well as a testing/validation sets and associated labels.
    The ratio size(y_validation)/size(y) is 1/k_fold.
    ex : k_fold=5 => 20% validation and 80% train.
    """
    k = 0
    k_indices = build_k_indices(y, k_fold, seed)
    x_validation = x[k_indices[k,:],:]
    y_validation = y[k_indices[k,:]]

    k_indices = np.delete(k_indices, k, axis=0).reshape(-1)
    x_train = x[k_indices, :]
    y_train = y[k_indices]

    return x_train, y_train, x_validation, y_validation
