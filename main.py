from implementations import *
from math_helpers import *
from io_helpers import *
from cross import *
import linear_regression as linreg
import logistic_regression as logreg
import numpy as np


if __name__ == '__main__':


    zero_and_one = False


#*****************************
#LOAD DATA
#*****************************

    print("Loading training data...")
    y_train, x_train, _ = load_csv_data("../data/train.csv", zero_and_one)

    #print("Loading testing data...")
    #_, x_test, ids = load_csv_data("../data/test.csv")







#*****************************
#DATA PREPROCESSING
#*****************************

    #replace column, per jet_num, that are undefinided for the corrsponding jet with 0 value
    #x_train, y_train = replace_column(x_train, y_train)

    #replace by median of the feature -999 value
    #x_train = remove999(x_train)

    #standardize by feature
    x_train, _, _ = standardize(x_train)
    #x_test, _, _ = standardize(x_test)

    #perform polynomialfeature expansion
    phi_train = build_poly(x_train, 2)
    #phi_test = build_poly(x_test, 2)







#*****************************
#PERFORM CROSS VALIDATION
#*****************************

    degrees = np.arange(1,3)
    lambdas = np.logspace(-5, 0, 20)
    #best_degree, best_lambda, best_rmse = best_degree_selection(y_train, x_train, degrees, 4, lambdas)


#*****************************
#CREATE VALIDATION SET
#*****************************

    phi_train, y_train, x_train_va, y_train_va = split_data(y_train, phi_train, 5, 2)




#*****************************
#DATA PROCESSING
#*****************************

    print("Training model...")

#initiate weigths
    initial_w = np.random.rand(phi_train.shape[1])
    #initial_w = np.zeros((phi_train.shape[1])
    #initial_w = np.ones((phi_train.shape[1])

    lambda_ = 0.01
    max_iters = 500
    gamma = 0.1


#least_squares_GD

    #w, loss = least_squares_GD(y_train, phi_train, initial_w, max_iters, gamma)

#least_squares_SGD

    #w, loss = least_squares_SGD(y_train, phi_train, initial_w, max_iters, gamma)

#least_squares

    w, loss = least_squares(y_train, phi_train)

#ridge_regression

    #w, loss = ridge_regression(y_train, phi_train, lambda_)

#logistic_regression

    #w, loss = logistic_regression(y_train, phi_train, initial_w, max_iters, gamma)

#reg_logistic_regression

    #w, loss = reg_logistic_regression(y_train, phi_train, lambda_, initial_w, max_iters, gamma)



#*****************************
#GET VALIDATION SET RESULT
#*****************************

    tr_accuracy = compute_accuracy(y_train, phi_train, w, zero_and_one)
    print("Training loss : " + str(loss) + ", accuracy : " + str(tr_accuracy))

    va_loss = logreg.compute_loss(y_train_va, x_train_va, w)
    #va_loss = linreg.compute_loss(y_train_va, x_train_va, w)

    va_accuracy = compute_accuracy(y_train_va, x_train_va, w, zero_and_one)
    print("Validation loss : " + str(va_loss) + ", accuracy : " + str(va_accuracy))


#*****************************
#PREDICTION
#*****************************


#least_squares and ridge_regression

    #y_test = phi_test@w

#logistic_regression and reg_logistic_regression

    #y_test = logreg.sigmoid(phi_test@w)


    #y_test = roundToPrediction(y_test, zero_and_one)



    """
    txs, ys, indices = separe_in_jet(x_train2, y_train2)
    txs_test, indices_test = separe_in_jet_test(x_test)
    y_test = np.zeros(x_test.shape[0])

    for i in range(4):

        x_train = txs[i]
        y_train2 = ys[i]

        x_train = remove999(x_train)
        x_train, _, _ = standardize(x_train)
        phi_train = build_poly(x_train, 2)
        initial_w = np.random.rand(phi_train.shape[1])
        w2, loss = logistic_regression(y_train2, phi_train, initial_w, 500, 0.1)

        x_test_i = txs_test[i]
        x_test_i = remove999(x_test_i)
        x_test_i, _, _ = standardize(x_test_i)
        x_test_i = build_poly(x_test_i, 2)
        y_test_i = logreg.sigmoid(x_test_i@w2)
        y_test_i = roundToPrediction(y_test_i)
        y_test[indices_test[i]] = y_test_i







    create_csv_submission(ids, y_test, "o.csv")


    tr_accuracy_jet = 0
    va_accuracy_jet = 0
    tr_loss_jet = 0
    va_loss_jet = 0
    tr_total_length = 0
    va_total_length = 0



    for i in range(4):

        x_train = txs[i]
        y_train = ys[i]

        x_train = remove999(x_train)
        x_train, _, _ = standardize(x_train)
        phi_train = build_poly(x_train, 2)
        x_train_tr_i, y_train_tr_i, x_train_va_i, y_train_va_i = split_data(y_train, phi_train, 5, 2)
        initial_w = np.random.rand(x_train_tr_i.shape[1])
        w2, loss = logistic_regression(y_train_tr_i, x_train_tr_i, initial_w, 500, 0.1)
        tr_loss_jet += loss
        tr_accuracy = compute_accuracy(y_train_tr_i, x_train_tr_i, w2)
        tr_accuracy_jet += (tr_accuracy*len(y_train_tr_i))
        va_loss = logreg.compute_loss(y_train_va_i, x_train_va_i, w2)
        va_loss_jet += va_loss
        va_accuracy = compute_accuracy(y_train_va_i, x_train_va_i, w2)
        va_accuracy_jet += (va_accuracy*len(y_train_va_i))
        tr_total_length += len(y_train_tr_i)
        va_total_length += len(y_train_va_i)
        print(str(i) + " : 4 JET Training loss : " + str(loss) + ", accuracy : " + str(tr_accuracy))
        print(str(i) + " : 4 JET Validation loss : " + str(va_loss_jet) + ", accuracy : " + str(va_accuracy))


    print("4 JET Training loss : " + str(tr_loss_jet) + ", accuracy : " + str(tr_accuracy_jet/tr_total_length))
    print("4 JET Validation loss : " + str(va_accuracy_jet) + ", accuracy : " + str(va_accuracy_jet/va_total_length))












    save_weights(w, "weights.txt")
    #create_csv_submission(ids, y_test, "o.csv")


    """

    print('\a')
