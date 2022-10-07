import numpy as np

def load_features(path_dataset):
    """Load all columns from data, droping the first two."""
    return np.genfromtxt(path_dataset, delimiter=",", skip_header=1)[:,2:]

def load_labels(path_dataset):
    """Load second column of data."""
    return np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=1)

if __name__ == '__main__':
    x_train = load_features("../data/train.csv")
    y_train = load_labels("../data/train.csv")
    x_test = load_features("../data/test.csv")

    print(x_train[:5,:])
    print(y_train[:5,:])
    print(x_test[:5,:])
