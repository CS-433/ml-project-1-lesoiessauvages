import csv
import numpy as np

# IO FUNCTIONS

def load_features(path_dataset):
    """Load all columns from data, droping the first two."""
    return np.genfromtxt(path_dataset, delimiter=",", skip_header=1)[:,2:]

def load_labels(path_dataset):
    """Load second column of data. Translates 'b' into -1 and 's' into 1."""
    labels = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=1, dtype=str)
    #labels = np.char.replace(labels, 'b', '-1')
    labels = np.char.replace(labels, 'b', '0')
    labels = np.char.replace(labels, 's', '1')

    return labels.astype(int)

def load_ids(path_dataset):
    """Load first column of data."""
    ids = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=0)
    return ids

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
