import csv
import numpy as np

def load_features(path_dataset):
    """Load all columns from data, droping the first two."""
    return np.genfromtxt(path_dataset, delimiter=",", skip_header=1)[:,2:]

def load_labels(path_dataset):
    """Load second column of data. Translates 'b' into -1 and 's' into 1.
    Also returns the ids."""
    labels = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=1, dtype=str)
    labels = np.char.replace(labels, 'b', '-1')
    labels = np.char.replace(labels, 's', '1')

    ids = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=0)
    return ids.astype(int), labels.astype(int)

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
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


if __name__ == '__main__':
    x_train = load_features("../data/train.csv")
    ids, y_train = load_labels("../data/train.csv")
    x_test = load_features("../data/test.csv")

    x_train, _, _ = standardize(x_train)
    x_test, _, _ = standardize(x_test)

    print(x_train[:5,:])
    print(y_train[:5])
    print(ids[:5])
    print(x_test[:5,:])

    create_csv_submission(np.arange(1000, 1005), np.array([1,1,-1,-1,1]), "o.csv")
