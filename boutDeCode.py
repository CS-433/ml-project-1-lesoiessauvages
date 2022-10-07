
#séparer le dataset en training et test
    rinds = np.random.permutation(N)
    X_tr = X[rinds[:int(N * split)]]
    y_tr = Y[rinds[:int(N * split)]]
    X_te = X[rinds[int(N * split):]]
    y_te = Y[rinds[int(N * split):]]

#basic polynomial expantion
    def expand_X(X, degree_of_expansion):
    """  Perform degree-d polynomial feature expansion of X,
        with bias but omitting interaction terms

    Args:
        X (np.array): data, shape (N, D).
        degree_of_expansion (int): The degree of the polynomial feature expansion.

    Returns:
        (np.array): Expanded data with shape (N, new_D),
                    where new_D is D*degree_of_expansion+1

    """

    expanded_X = np.ones((X.shape[0],1))
    for idx in range(1,degree_of_expansion+1):
        expanded_X = np.hstack((expanded_X, X**idx))
    return expanded_X


#example
    X_toy = np.arange(15).reshape((5,3))
    mean = np.mean(X_toy, axis=0, keepdims=True)
    std = np.std(X_toy, axis=0, keepdims=True)
    X_toy_norm = normalize_X(X_toy, mean, std)
    expanded_X_toy = expand_X(X_toy_norm, degree_of_expansion)


#cross validation

def do_cross_validation(k, k_fold_ind, X, Y, degree_of_expansion=1, lmda=0.1):
    # use one split as val
    val_ind = k_fold_ind[k]
    # use k-1 split to train
    train_splits = [i for i in range(k_fold_ind.shape[0]) if i is not k]
    train_ind = k_fold_ind[train_splits,:].reshape(-1)

    #Get train and val
    cv_X_tr = X[train_ind,:]
    cv_Y_tr = Y[train_ind]
    cv_X_val = X[val_ind,:]
    cv_Y_val = Y[val_ind]

    #expand and normalize for degree d
    cv_X_tr_poly = expand_X(cv_X_tr, degree_of_expansion)
    cv_X_val_poly = expand_X(cv_X_val, degree_of_expansion)

    mean = np.mean(cv_X_tr_poly[:, 1:], axis=0)
    std = np.std(cv_X_tr_poly[:, 1:], axis=0)

    norm_cv_X_tr_poly = np.ones_like(cv_X_tr_poly)
    norm_cv_X_val_poly = np.ones_like(cv_X_val_poly)

    norm_cv_X_tr_poly[:, 1:] = normalize_X(cv_X_tr_poly[:, 1:], mean, std)
    norm_cv_X_val_poly[:, 1:] = normalize_X(cv_X_val_poly[:, 1:], mean, std)

    #fit on train set
    w = get_w_analytical(cv_X_tr_poly, cv_Y_tr, lmda)
    pred = cv_X_val_poly@w

    #get loss for val
    loss_test = metric_mse(pred, cv_Y_val)
    return loss_test


# Function to split data indices
# num_examples: total samples in the dataset
# k_fold: number fold of CV
# returns: array of shuffled indices with shape (k_fold, num_examples//k_fold)
def fold_indices(num_examples,k_fold):
    ind = np.arange(num_examples)
    split_size = num_examples//k_fold

    #important to shuffle your data
    np.random.shuffle(ind)

    k_fold_indices = []
    # Generate k_fold set of indices
    k_fold_indices = [ind[k*split_size:(k+1)*split_size] for k in range(k_fold)]

    return np.array(k_fold_indices)

'''Plotting Heatmap for CV results'''
def plot_cv_result(grid_val,grid_search_lambda,grid_search_degree):
    plt.figure(figsize=(8,10))
    plt.imshow(grid_val)
    plt.colorbar()
    plt.xticks(np.arange(len(grid_search_degree)), grid_search_degree, rotation=20)
    plt.yticks(np.arange(len(grid_search_lambda)), grid_search_lambda, rotation=20)
    plt.xlabel('degree')
    plt.ylabel('lambda')
    plt.title('Val Loss for different lambda and degree')
    plt.show()



'''
Grid Search Function
params:{'param1':[1,2,..,4],'param2':[6,7]} dictionary of search params
k_fold: fold for CV to be done
fold_ind: splits of training set
function: implementation of model should return a loss or score
X,Y: training examples
'''
def grid_search_cv(params,k_fold,fold_ind,function,X,Y):

    #might mess up with dictionary order
    param_grid = ParameterGrid(params)
    #save the values for the combination of hyperparameters
    grid_val = np.zeros(len(param_grid))
    grid_val_std = np.zeros(len(param_grid))

    for i, p in enumerate(param_grid):
        #print('Evaluating for {} ...'.format(p))
        loss = np.zeros(k_fold)
        for k in range(k_fold):
            loss[k] = function(k,fold_ind,X,Y,**p)
        grid_val[i] = np.mean(loss)
        grid_val_std[i] = np.std(loss)

    # reshape in the proper dimension of search space
    if len(params.keys())>1:
        search_dim = tuple([len(p) for _,p in params.items()])
        grid_val = grid_val.reshape(search_dim)
        grid_val_std = grid_val_std.reshape(search_dim)

    return grid_val, grid_val_std

#execution

    #list of lambda values to try.. use np.logspace
    search_lambda = np.logspace(-2,1,num=10)
    #list of degrees
    search_degree = np.arange(1,15,1)

    params = {'degree_of_expansion':search_degree,'lmda':search_lambda,}

    k_fold = 3
    fold_ind = fold_indices(X_tr.shape[0],k_fold)
    #call to the grid search function
    grid_val, grid_val_std = grid_search_cv(params,k_fold,fold_ind,do_cross_validation,X_tr,y_tr)


    #get the best validation score
    best_score = np.min(grid_val)
    print('Best val score {}'.format(best_score))

    #get degree which gives best score
    d,l= np.unravel_index(np.argmin(grid_val, axis=None),grid_val.shape)
    best_degree = search_degree[d]
    best_lambda = search_lambda[l]
    print('Best score achieved using degree:{} and lambda:{}'.format(best_degree,best_lambda))

    from helper import plot_cv_result
    plot_cv_result(np.log((grid_val.T)),search_lambda,search_degree)
