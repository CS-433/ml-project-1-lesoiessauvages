from implementations import *
from math_helpers import *
from io_helpers import *
from cross import *
import linear_regression as linreg
import logistic_regression as logreg
import numpy as np


if __name__ == '__main__':
    print("Loading training data...")
    y_train, x_train, _ = load_csv_data("../data/train.csv")
    print("Loading testing data...")
    _, x_test, ids = load_csv_data("../data/test.csv")

    x_train, _, _ = standardize(x_train)
    x_test, _, _ = standardize(x_test)

    phi_train = build_poly(x_train, 2)
    #x_train_tr, y_train_tr, x_train_va, y_train_va = split_data(y_train, phi_train, 5)
    phi_test = build_poly(x_test, 2)

    # print(phi_train[:5,:])
    # print(y_train[:5])
    # print(ids[:5])
    # print(x_test[:5,:])
    # print(np.min(phi_train))

    #degrees = np.arange(1,6)
    #lambdas = np.logspace(-5, 2, 20)
    #best_degree, best_lambda, best_rmse = best_degree_selection(y_train, x_train, degrees, 4, lambdas)


    print("Training model...")

    ##w, loss = logistic_regression(y_train, phi_train, np.zeros(30), 1000, 0.000002)
    ##w, loss = ridge_regression(y_train, phi_train, 10)
    ##w, loss = least_squares_SGD(y_train, phi_train, np.ones(120), 1000, 0.001)
    ##w, loss = least_squares_SGD(y_train, x_train, np.ones(30), 1000, 0.001)
    # w, loss = logistic_regression(y_train_tr, x_train_tr, np.ones(x_train_tr.shape[1]), 5000, 0.1)
    w, loss = logistic_regression(y_train, phi_train, np.ones(phi_train.shape[1]), 3000, 0.1)

    #phi_train = build_poly(x_train, best_degree)
    #phi_test = build_poly(x_test, best_degree)
    #w, loss = ridge_regression(y_train, phi_train, best_lambda)

    # y_test = logreg.sigmoid(phi_test@w)
    #y_test = x_test@w
    ##y_test = phi_test@w



    y_test = logreg.sigmoid(phi_test@w)

    # print(y_test)
    y_test = roundToPrediction(y_test)


    # tr_accuracy = compute_accuracy(y_train_tr, x_train_tr, w)
    # print("Training loss : " + str(loss) + ", accuracy : " + str(tr_accuracy))
    # va_loss = logreg.compute_loss(y_train_va, x_train_va, w)
    # va_accuracy = compute_accuracy(y_train_va, x_train_va, w)
    # print("Validation loss : " + str(va_loss) + ", accuracy : " + str(va_accuracy))

    save_weights(w, "weights.txt")
    create_csv_submission(ids, y_test, "o.csv")
