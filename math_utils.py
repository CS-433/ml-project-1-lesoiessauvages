import numpy as np
import math as math
import linear_regression as linreg
import logistic_regression as logreg


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
    data_size = y.shape[0]

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


def build_poly(x, degree, k_corr=0):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    + multiplication of k_corr least corellated features."""
    output = np.ones(x.shape[0])
    for i in range(1, degree + 1):
        output = np.c_[output, x ** i]

    R1 = np.abs(np.corrcoef(x.T))

    if k_corr != 0:

        R1 = np.where(np.tril(R1) == 0, R1, 1)
        argsort = np.argpartition(R1, k_corr, axis=None)

        for i in range(k_corr):
            a = int(np.floor(argsort[i] / x.shape[1]))
            b = argsort[i] % x.shape[1]
            output = np.c_[output, x[:, a] * x[:, b]]

    return output


def normalize(x):
    """normalize the original data set column-wise."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def roundToPrediction(a, logistic_model):
    """Step function to clip output."""

    if logistic_model:
        a[a < 0.5] = -1
        a[a >= 0.5] = 1
    else:
        a[a < 0] = -1
        a[a >= 0] = 1
    return a


def roundToPredictionForAccuracy(a, logistic_model):
    """Step function to clip output."""

    if logistic_model:
        a[a < 0.5] = 0
        a[a >= 0.5] = 1
    else:
        a[a < 0] = -1
        a[a >= 0] = 1
    return a


def compute_accuracy(y, tx, w, logistic_model):

    if logistic_model:
        pred = logreg.sigmoid(tx @ w)
    else:
        pred = tx @ w
    pred = roundToPredictionForAccuracy(pred, logistic_model)
    accuracy = np.count_nonzero(pred == y) / y.shape[0]
    return accuracy


def replace_all_999_with_0(tx):

    a = tx
    a[a == -999] = 0

    return a


def replace_all_999_with_mean(tx):

    a = tx
    a[a == -999] = math.nan
    col_mean = np.nanmean(a, axis=0)
    # col_mean = np.zeros(col_mean.shape)
    # Find indices that you need to replace
    inds = np.where(np.isnan(a))
    # Place column means in the indices. Align the arrays using take
    a[inds] = np.take(col_mean, inds[1])

    return a


def replace_all_999_with_median(tx):

    a = tx
    a[a == -999] = math.nan
    col_mean = np.nanmedian(a, axis=0)
    # col_mean = np.zeros(col_mean.shape)
    # Find indices that you need to replace
    inds = np.where(np.isnan(a))
    # Place column means in the indices. Align the arrays using take
    a[inds] = np.take(col_mean, inds[1])

    return a


def split_in_jets(tx, y):

    jets = tx[:, 22]

    tx_0 = tx[jets == 0]
    indices_0 = np.asarray(jets == 0).nonzero()
    tx_0 = np.delete(tx_0, [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29], 1)

    tx_1 = tx[jets == 1]
    indices_1 = np.asarray(jets == 1).nonzero()
    tx_1 = np.delete(tx_1, [4, 5, 6, 12, 22, 26, 27, 28], 1)

    tx_2 = tx[jets == 2]
    indices_2 = np.asarray(jets == 2).nonzero()
    tx_2 = np.delete(tx_2, 22, 1)

    tx_3 = tx[jets == 3]
    indices_3 = np.asarray(jets == 3).nonzero()
    tx_3 = np.delete(tx_3, 22, 1)

    y_0 = y[jets == 0]
    y_1 = y[jets == 1]
    y_2 = y[jets == 2]
    y_3 = y[jets == 3]
    return (
        [tx_0, tx_1, tx_2, tx_3],
        [y_0, y_1, y_2, y_3],
        [indices_0, indices_1, indices_2, indices_3],
    )


def split_in_jets_test(tx):

    jets = tx[:, 22]

    tx_0 = tx[jets == 0]
    indices_0 = np.asarray(jets == 0).nonzero()
    tx_0 = np.delete(tx_0, [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29], 1)

    tx_1 = tx[jets == 1]
    indices_1 = np.asarray(jets == 1).nonzero()
    tx_1 = np.delete(tx_1, [4, 5, 6, 12, 22, 26, 27, 28], 1)

    tx_2 = tx[jets == 2]
    indices_2 = np.asarray(jets == 2).nonzero()
    tx_2 = np.delete(tx_2, 22, 1)

    tx_3 = tx[jets == 3]
    indices_3 = np.asarray(jets == 3).nonzero()
    tx_3 = np.delete(tx_3, 22, 1)

    return [tx_0, tx_1, tx_2, tx_3], [indices_0, indices_1, indices_2, indices_3]


def replace_undefined_with_0(tx, y):
    """Replaces undefined features by 0, according to feature PRI_jet_num
    """
    jets = tx[:, 22]

    tx_0 = tx[jets == 0]
    tx_0[:, [4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29]] = 0
    indices_0 = np.asarray(jets == 0).nonzero()

    tx_1 = tx[jets == 1]
    tx_1[:, [4, 5, 6, 12, 26, 27, 28]] = 0
    indices_1 = np.asarray(jets == 1).nonzero()

    tx_2 = tx[jets == 2]
    indices_2 = np.asarray(jets == 2).nonzero()

    tx_3 = tx[jets == 3]
    indices_3 = np.asarray(jets == 3).nonzero()

    y_0 = y[jets == 0]
    y_1 = y[jets == 1]
    y_2 = y[jets == 2]
    y_3 = y[jets == 3]

    new_tx = np.concatenate((tx_0, tx_1, tx_2, tx_3), axis=0)
    new_ty = np.concatenate((y_0, y_1, y_2, y_3), axis=0)

    indices = np.concatenate((indices_0, indices_1, indices_2, indices_3), axis=1)

    return new_tx, new_ty, indices
