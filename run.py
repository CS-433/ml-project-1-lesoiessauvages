from implementations import *
from math_utils import *
from helpers import *
from cross import *
import linear_regression as linreg
import logistic_regression as logreg
import numpy as np






def sigmoid2(t):
    """apply sigmoid function on t.
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """
    return 1.0 / (1 + np.exp(-t))


def compute_loss2(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss

    """
    #print("begin compute loss")
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    ### CLASSICAL VERSION OF THE FORMULA
    pred = sigmoid2(tx@w)

    ### VERSION WITH EPSILON TO AVOID INVALID VALUES
    epsilon = 1e-7
    loss = y.T@np.log(pred + epsilon) + (1 - y).T@np.log(1 - pred + epsilon)

    return 1/2*(-loss.item())/y.shape[0]


def compute_gradient2(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    pred = sigmoid2(tx@w)
    grad = np.mean((pred - y)*tx.T, 1)
    return grad



def logistic_regression2(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent or SGD (y ∈ {0, 1}).
    Returns :
        w : Last weight vector
        loss : coresponding loss
    """
    w = initial_w

    for iter in range(max_iters):
        grad = compute_gradient2(y, tx, w)
        loss = compute_loss2(y, tx, w)
        w -= gamma * grad
        if(iter%50==0):
            print("iter = " + str(iter) + " : " + str(loss))
    return w, loss





if __name__ == '__main__':

    logistic_model = True


#*****************************
#LOAD DATA
#*****************************

    print("Loading training data...")
    y_train, x_train, _ = load_csv_data("../data/train.csv", logistic_model)

    print("Loading testing data...")
    _, x_test, ids = load_csv_data("../data/test.csv", logistic_model)









#*****************************
#DATA PREPROCESSING
#*****************************

    degree = 2

    #replace column, per jet_num, that are undefinided for the corrsponding jet with 0 value
    #x_train, y_train, _ = replace_column(x_train, y_train)
    #x_test, _ , ids = replace_column(x_test, zeros((x_test.shape[0])))

    #replace by median of the feature -999 value
    #x_train = remove999_with0(x_train)
    #x_test = remove999_with0(x_test)

    #x_train = remove999_withmean(x_train)
    #x_test = remove999_withmean(x_test)

    #x_train = remove999_withmedian(x_train)
    #x_test = remove999_withmedian(x_test)

    #normalize by feature
    #x_train, _, _ = normalize(x_train)
    #x_test, _, _ = normalize(x_test)







    #perform polynomialfeature expansion
    #phi_train = build_poly(x_train, degree)
    #phi_test = build_poly(x_test, degree)

    #phi_train = x_train
    #phi_test = x_test






#*****************************
#PERFORM CROSS VALIDATION
#*****************************

    degrees = np.arange(1,4)
    lambdas = np.logspace(-5, 0, 10)
    gammas = np.logspace(-6, 1, 10)
    #best_degree, best_lambda, best_rmse = best_degree_lambda_selection(y_train, x_train, degrees, 5, lambdas)
    #best_degree, best_lambda, best_gamma, best_rmse = best_degree_lambda_gamma_selection(y_train, x_train, degrees, 5, lambdas, gammas)




#*****************************
#CREATE VALIDATION SET
#*****************************

    #phi_train, y_train, x_train_va, y_train_va = split_data(y_train, phi_train, 5, 1)




#*****************************
#DATA PROCESSING
#*****************************

    print("Training model...")

#initiate weigths
    np.random.seed(1)
    #initial_w = np.random.rand(phi_train.shape[1])
    #initial_w = np.zeros((phi_train.shape[1]))
    #initial_w = np.ones((phi_train.shape[1]))

    lambda_ = 0.003
    max_iter = 5000
    gamma = 0.1
    k_corr = 15





#least_squares_GD

    #w, loss = least_squares_GD(y_train, phi_train, initial_w, max_iter, gamma)

#least_squares_SGD

    #w, loss = least_squares_SGD(y_train, phi_train, initial_w, max_iter, gamma)

#least_squares

    #w, loss = least_squares(y_train, phi_train)

#ridge_regression

    #w, loss = ridge_regression(y_train, phi_train, lambda_)

#logistic_regression

    #w, loss = logistic_regression(y_train, phi_train, initial_w, max_iter, gamma)

#reg_logistic_regression

    #w, loss = reg_logistic_regression(y_train, phi_train, lambda_, initial_w, max_iter, gamma)



#*****************************
#GET VALIDATION SET RESULT
#*****************************

    # tr_accuracy = compute_accuracy(y_train, phi_train, w, logistic_model)
    # print("Training loss : " + str(loss) + ", accuracy : " + str(tr_accuracy))
    #
    # va_loss = logreg.compute_loss(y_train_va, x_train_va, w)
    # #va_loss = linreg.compute_loss(y_train_va, x_train_va, w)
    #
    # va_accuracy = compute_accuracy(y_train_va, x_train_va, w, logistic_model)
    # print("thresh 10 ⁻5,  Validation loss : " + str(va_loss) + ", accuracy : " + str(va_accuracy))



#*****************************
#SEPERATE IN FOUR JET FOR SUBMISSION
#*****************************

    x_train_jet, y_train_jet, indices = separe_in_jet(x_train, y_train)
    x_test_jet, indices_jet = separe_in_jet_test(x_test)
    y_test = np.zeros(x_test.shape[0])

    for i in range(4):

        x_train = x_train_jet[i]
        y_train = y_train_jet[i]

        x_train = remove999_withmedian(x_train)
        x_train, _, _ = normalize(x_train)
        phi_train = build_poly(x_train, degree, k_corr)
        initial_w = np.zeros(phi_train.shape[1])
        w, loss = logistic_regression2(y_train, phi_train, initial_w, max_iter, gamma)

        x_test_i = x_test_jet[i]
        x_test_i = remove999_withmedian(x_test_i)
        x_test_i, _, _ = normalize(x_test_i)
        x_test_i = build_poly(x_test_i, degree, k_corr)
        y_test_i = logreg.sigmoid(x_test_i@w)
        y_test_i = roundToPrediction(y_test_i, logistic_model)
        y_test[indices_jet[i]] = y_test_i




#*****************************
#SEPERATE IN FOR JET FOR VALIDATION
#*****************************

    """
    tr_accuracy_jet = 0
    va_accuracy_jet = 0
    tr_loss_jet = 0
    va_loss_jet = 0
    tr_total_length = 0
    va_total_length = 0

    x_train_jet, y_train_jet, indices = separe_in_jet(x_train, y_train)

    for i in range(4):

        x_train = x_train_jet[i]
        y_train = y_train_jet[i]

        x_train = remove999_withmedian(x_train)
        x_train, _, _ = normalize(x_train)

        phi_train = build_poly(x_train, 2, 200)
        #phi_train = x_train
        initial_w = np.zeros((phi_train.shape[1]))


        x_train_tr_i, y_train_tr_i, x_train_va_i, y_train_va_i = split_data(y_train, phi_train, 5, 1)

        w, tr_loss = reg_logistic_regression(y_train_tr_i, x_train_tr_i, lambda_, initial_w, max_iter, gamma)
        tr_loss_jet += (tr_loss*len(y_train_tr_i))
        tr_accuracy = compute_accuracy(y_train_tr_i, x_train_tr_i, w, logistic_model)
        tr_accuracy_jet += (tr_accuracy*len(y_train_tr_i))

        va_loss = logreg.compute_loss(y_train_va_i, x_train_va_i, w)
        va_loss_jet += (va_loss*len(y_train_va_i))
        va_accuracy = compute_accuracy(y_train_va_i, x_train_va_i, w, logistic_model)
        va_accuracy_jet += (va_accuracy*len(y_train_va_i))

        tr_total_length += len(y_train_tr_i)
        va_total_length += len(y_train_va_i)
        print(str(i) + " : 4 JET Training loss : " + str(tr_loss) + ", accuracy : " + str(tr_accuracy))
        print(str(i) + " : 4 JET Validation loss : " + str(va_loss) + ", accuracy : " + str(va_accuracy))


    print("4 JET Training loss : " + str(tr_loss_jet/tr_total_length) + ", accuracy : " + str(tr_accuracy_jet/tr_total_length))
    print("4 JET Validation loss : " + str(va_loss_jet/va_total_length) + ", accuracy : " + str(va_accuracy_jet/va_total_length))
    """



#*****************************
#PREDICTION
#*****************************


#least_squares and ridge_regression

    #y_test = phi_test@w

#logistic_regression and reg_logistic_regression

    #y_test = logreg.sigmoid(phi_test@w)


    #y_test = roundToPrediction(y_test, logistic_model)



#*****************************
#WRITE RESULT
#*****************************

    #save_weights(w, "weights.txt")
    create_csv_submission(ids, y_test, "o.csv")


    print('\a')
