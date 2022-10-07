def load_data(path_dataset):
    """Load data."""
    return data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1])

if __name__ == '__main__':
    train = load_data("../data/train.csv")
    test = load_data("../data/test.csv")
    x_train = train[:,2:]
    y_train = train[]
    x_test = test[:,2:]

    print(x_train[:5,:])
    print(y_train[:5,:])
    print(x_test[:5,:])
