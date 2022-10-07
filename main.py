import numpy as np

def load_features(path_dataset):
    """Load all columns from data, droping the first two."""
    return np.genfromtxt(path_dataset, delimiter=",", skip_header=1)[:,2:]

def load_labels(path_dataset):
    """Load second column of data. Translates 'b' into -1 and 's' into 1"""
    labels = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=1, dtype=str)
    labels = np.char.replace(labels, 'b', '-1')
    labels = np.char.replace(labels, 's', '1')
    return labels.astype(int)

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


if __name__ == '__main__':
    x_train = load_features("../data/train.csv")
    y_train = load_labels("../data/train.csv")
    x_test = load_features("../data/test.csv")

    x_train, _, _ = standardize(x_train)
    x_test, _, _ = standardize(x_test)

    print(x_train[:5,:])
    print(y_train[:5])
    print(x_test[:5,:])
