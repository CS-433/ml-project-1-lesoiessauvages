from implementations import *
import csv
import numpy as np

def load_features(path_dataset):
    """Load all columns from data, droping the first two."""
    return np.genfromtxt(path_dataset, delimiter=",", skip_header=1)[:,2:]

def load_labels(path_dataset):
    """Load second column of data. Translates 'b' into -1 and 's' into 1."""
    labels = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=1, dtype=str)
    labels = np.char.replace(labels, 'b', '-1')
    labels = np.char.replace(labels, 's', '1')

    return labels.astype(int)

def load_ids(path_dataset):
    """Load first column of data."""
    ids = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=0)
    return ids

# TODO: standardize column-wise
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})

def roundToPrediction(a):
    a[a < 0] = -1
    a[a >= 0] = 1
    return a

if __name__ == '__main__':
    x_train = load_features("../data/train.csv")
    y_train = load_labels("../data/train.csv")
    x_test = load_features("../data/test.csv")
    ids = load_ids("../data/test.csv")

    x_train, _, _ = standardize(x_train)
    x_test, _, _ = standardize(x_test)

    # print(x_train[:5,:])
    # print(y_train[:5])
    # print(ids[:5])
    # print(x_test[:5,:])

    ##w, loss = least_squares_SGD(y_train, x_train, np.ones(30), 1000, 0.001)
    w, loss = least_squares(y_train, x_train)

    y_test = x_test@w
    y_test = roundToPrediction(y_test)

    print(loss)


    create_csv_submission(ids, y_test, "o.csv")
