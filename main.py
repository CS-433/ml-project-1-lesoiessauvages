from implementations import *
from math_utils import *
from helpers import *
from cross import *
import linear_regression as linreg
import logistic_regression as logreg
import numpy as np


def logistic_regression2(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1}).
    Calls compute_loss with extra variable epsilon to avoid invalid values in logarithms. Cf. README and report.
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """
    w = initial_w
    loss = logreg.compute_loss(y, tx, w)

    for iter in range(max_iters):
        grad = logreg.compute_gradient(y, tx, w)
        w -= gamma * grad
        loss = logreg.compute_loss(y, tx, w, True)
        if iter % 50 == 0:
            print("iter = " + str(iter) + " : " + str(loss))
    return w, loss


if __name__ == "__main__":

    ### Needs to be set to true in case of working with logistic regression.
    ### maps y to {0,1} or {-1,1}
    logistic_model = True

    # *****************************
    # LOAD DATA
    # *****************************

    print("Loading training data...")
    y_train, x_train, _ = load_csv_data("../data/train.csv")

    if logistic_model:
        y_train[np.where(y_train == -1)] = 0

    print("Loading testing data...")
    _, x_test, ids = load_csv_data("../data/test.csv")

    # *****************************
    # DATA PREPROCESSING
    # *****************************

    degree = 2

    # replace column, per jet_num, that are undefinided for the corrsponding jet with 0 value
    # x_train, y_train, _ = replace_undefined_with_0(x_train, y_train)
    # x_test, _ , ids = replace_undefined_with_0(x_test, zeros((x_test.shape[0])))

    # replace by median of the feature -999 value
    # x_train = replace_all_999_with_0(x_train)
    # x_test = replace_all_999_with_0(x_test)

    # x_train = replace_all_999_with_mean(x_train)
    # x_test = replace_all_999_with_mean(x_test)

    # x_train = replace_all_999_with_median(x_train)
    # x_test = replace_all_999_with_median(x_test)

    # normalize by feature
    # x_train, _, _ = normalize(x_train)
    # x_test, _, _ = normalize(x_test)

    ###perform polynomialfeature expansion
    # phi_train = build_poly(x_train, degree)
    # phi_test = build_poly(x_test, degree)

    # phi_train = x_train
    # phi_test = x_test

    # *****************************
    # PERFORM CROSS VALIDATION
    # *****************************

    degrees = np.arange(1, 4)
    lambdas = np.logspace(-5, 0, 10)
    gammas = np.logspace(-6, 1, 10)
    # best_degree, best_lambda, best_rmse = best_degree_lambda_selection(y_train, x_train, degrees, 5, lambdas)
    # best_degree, best_lambda, best_gamma, best_rmse = best_degree_lambda_gamma_selection(y_train, x_train, degrees, 5, lambdas, gammas)

    # *****************************
    # CREATE VALIDATION SET
    # *****************************

    # phi_train, y_train, x_train_va, y_train_va = split_data(y_train, phi_train, 5, 1)

    # *****************************
    # DATA PROCESSING
    # *****************************

    print("Training model...")

    # initiate weigths
    np.random.seed(1)
    # initial_w = np.random.rand(phi_train.shape[1])
    # initial_w = np.zeros((phi_train.shape[1]))
    # initial_w = np.ones((phi_train.shape[1]))

    lambda_ = 0.003
    max_iter = 25000
    gamma = 0.2
    k_corr = 15

    # mean_squared_error_gd

    # w, loss = mean_squared_error_gd(y_train, phi_train, initial_w, max_iter, gamma)

    # mean_squared_error_sgd

    # w, loss = mean_squared_error_sgd(y_train, phi_train, initial_w, max_iter, gamma)

    # least_squares

    # w, loss = least_squares(y_train, phi_train)

    # ridge_regression

    # w, loss = ridge_regression(y_train, phi_train, lambda_)

    # logistic_regression

    # w, loss = logistic_regression(y_train, phi_train, initial_w, max_iter, gamma)

    # reg_logistic_regression

    # w, loss = reg_logistic_regression(y_train, phi_train, lambda_, initial_w, max_iter, gamma)

    # *****************************
    # GET VALIDATION SET RESULT
    # *****************************

    # tr_accuracy = compute_accuracy(y_train, phi_train, w, logistic_model)
    # print("Training loss : " + str(loss) + ", accuracy : " + str(tr_accuracy))
    #
    # va_loss = logreg.compute_loss(y_train_va, x_train_va, w)
    # # va_loss = linreg.compute_loss(y_train_va, x_train_va, w)
    #
    # va_accuracy = compute_accuracy(y_train_va, x_train_va, w, logistic_model)
    # print(
    #     "Validation loss : "
    #     + str(va_loss)
    #     + ", accuracy : "
    #     + str(va_accuracy)
    # )

    # *****************************
    # SEPERATE IN FOUR JET FOR SUBMISSION
    # *****************************

    x_train_jet, y_train_jet, indices = split_in_jets(x_train, y_train)
    x_test_jet, indices_jet = split_in_jets_test(x_test)
    y_test = np.zeros(x_test.shape[0])

    for i in range(4):

        x_train = x_train_jet[i]
        y_train = y_train_jet[i]

        x_train = replace_all_999_with_median(x_train)
        x_train, _, _ = normalize(x_train)
        phi_train = build_poly(x_train, degree, k_corr)
        initial_w = np.zeros(phi_train.shape[1])
        w, loss = logistic_regression2(y_train, phi_train, initial_w, max_iter, gamma)

        x_test_i = x_test_jet[i]
        x_test_i = replace_all_999_with_median(x_test_i)
        x_test_i, _, _ = normalize(x_test_i)
        x_test_i = build_poly(x_test_i, degree, k_corr)
        y_test_i = logreg.sigmoid(x_test_i @ w)
        y_test_i = roundToPrediction(y_test_i, logistic_model)
        y_test[indices_jet[i]] = y_test_i

    # *****************************
    # SEPERATE IN FOUR JET FOR VALIDATION
    # *****************************

    # tr_accuracy_jet = 0
    # va_accuracy_jet = 0
    # tr_loss_jet = 0
    # va_loss_jet = 0
    # tr_total_length = 0
    # va_total_length = 0
    #
    # x_train_jet, y_train_jet, indices = split_in_jets(x_train, y_train)
    #
    # for i in range(4):
    #
    #     x_train = x_train_jet[i]
    #     y_train = y_train_jet[i]
    #
    #     x_train = replace_all_999_with_median(x_train)
    #     x_train, _, _ = normalize(x_train)
    #
    #     phi_train = build_poly(x_train, 2)
    #     #phi_train = x_train
    #     initial_w = np.ones((phi_train.shape[1]))
    #
    #
    #     x_train_tr_i, y_train_tr_i, x_train_va_i, y_train_va_i = split_data(y_train, phi_train, 5, 1)
    #
    #     w, tr_loss = logistic_regression(y_train_tr_i, x_train_tr_i, initial_w, 20000,0.0006)
    #     tr_loss_jet += (tr_loss*len(y_train_tr_i))
    #     tr_accuracy = compute_accuracy(y_train_tr_i, x_train_tr_i, w, logistic_model)
    #     tr_accuracy_jet += (tr_accuracy*len(y_train_tr_i))
    #
    #     va_loss = logreg.compute_loss(y_train_va_i, x_train_va_i, w)
    #     va_loss_jet += (va_loss*len(y_train_va_i))
    #     va_accuracy = compute_accuracy(y_train_va_i, x_train_va_i, w, logistic_model)
    #     va_accuracy_jet += (va_accuracy*len(y_train_va_i))
    #
    #     tr_total_length += len(y_train_tr_i)
    #     va_total_length += len(y_train_va_i)
    #     print(str(i) + " : 4 JET Training loss : " + str(tr_loss) + ", accuracy : " + str(tr_accuracy))
    #     print(str(i) + " : 4 JET Validation loss : " + str(va_loss) + ", accuracy : " + str(va_accuracy))
    #
    #
    # print("4 JET Training loss : " + str(tr_loss_jet/tr_total_length) + ", accuracy : " + str(tr_accuracy_jet/tr_total_length))
    # print("4 JET Validation loss : " + str(va_loss_jet/va_total_length) + ", accuracy : " + str(va_accuracy_jet/va_total_length))

    # *****************************
    # PREDICTION
    # *****************************

    # least_squares and ridge_regression

    # y_test = phi_test@w

    # logistic_regression and reg_logistic_regression

    # y_test = logreg.sigmoid(phi_test@w)

    # y_test = roundToPrediction(y_test, logistic_model)

    # *****************************
    # WRITE RESULT
    # *****************************

    create_csv_submission(ids, y_test, "o.csv")

    print("\a")
