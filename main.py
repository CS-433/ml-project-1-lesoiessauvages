from implementations import *
from math_helpers import *
from io_helpers import *
import numpy as np


if __name__ == '__main__':
    x_train = load_features("../data/train.csv")
    y_train = load_labels("../data/train.csv")
    x_test = load_features("../data/test.csv")
    ids = load_ids("../data/test.csv")

    x_train, _, _ = standardize(x_train)
    x_test, _, _ = standardize(x_test)

    phi_train = build_poly(x_train, 6)
    phi_test = build_poly(x_test, 6)

    # print(x_train[:5,:])
    # print(y_train[:5])
    # print(ids[:5])
    # print(x_test[:5,:])

    w, loss = ridge_regression(y_train, phi_train, 10)
    ##w, loss = least_squares_SGD(y_train, phi_train, np.ones(120), 1000, 0.001)
    ##w, loss = least_squares_SGD(y_train, x_train, np.ones(30), 1000, 0.001)
    ##w, loss = least_squares(y_train, x_train)

    ##y_test = x_test@w
    y_test = phi_test@w
    y_test = roundToPrediction(y_test)

    print(loss)


    create_csv_submission(ids, y_test, "o.csv")
