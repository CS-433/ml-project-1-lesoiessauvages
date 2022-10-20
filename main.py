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

    phi_train = build_poly(x_train, 7)
    phi_train_tr, y_train_tr, phi_train_va, y_train_va = split_data(y_train, phi_train, 5)
    phi_test = build_poly(x_test, 7)

    # print(x_train[:5,:])
    # print(y_train[:5])
    # print(ids[:5])
    # print(x_test[:5,:])

    #degrees = np.arange(1,6)
    #lambdas = np.logspace(-5, 2, 20)
    #best_degree, best_lambda, best_rmse = best_degree_selection(y_train, x_train, degrees, 4, lambdas)


    print("Training model...")

    ##w, loss = logistic_regression(y_train, phi_train, np.zeros(30), 1000, 0.000002)
    ##w, loss = ridge_regression(y_train, phi_train, 10)
    ##w, loss = least_squares_SGD(y_train, phi_train, np.ones(120), 1000, 0.001)
    ##w, loss = least_squares_SGD(y_train, x_train, np.ones(30), 1000, 0.001)
    w, loss = least_squares(y_train_tr, phi_train_tr)

    #phi_train = build_poly(x_train, best_degree)
    #phi_test = build_poly(x_test, best_degree)
    #w, loss = ridge_regression(y_train, phi_train, best_lambda)

    y_test = phi_test@w
    #y_test = x_test@w
    ##y_test = phi_test@w



    #y_test = sigmoid(y_test)

    print(y_test)
    y_test = roundToPrediction(y_test)


    print("Training loss : " + str(loss))
    print("Validation loss : " + str(linreg.compute_loss(y_train_va, phi_train_va, w)))

    save_weights(w, "weights.txt")
    create_csv_submission(ids, y_test, "o.csv")
