# ml-project-1-lesoiessauvages

## The project

Detecting the Higgs' boson is possible through its _decay signature_. In this lab, we will apply binary classification to a dataset in order to determine whether an observation shows the presence of the Higgs' boson or not. Several models and methods were tried and compared in order to find the most accurate one.

We are working on the dataset of observations in the LHC. It contains a training set of 250000 datapoints, each of 30 features, labelled with 1 or -1 for _signal_ or _background_ respectively. The test set is larger, it contains 568238 unlabelled datapoints.

We tried iteratively the 6 main models, along with various data preprocessing. Cf. our report for more detailed information.

## Description

- `main.py` : Main development file. Contains a lot of lines than can be uncommented to run different setup. Contains a boolean variable `logistic_model` that must be set to true if working with logistic regression.

- `implementations.py` : Contains the 6 mandatory implementations : 
  - Linear model : stochastic gradient descent
  - Linear model : gradient descent
  - Linear model : least squares method
  - Linear model : ridge regression
  - Logistic regression (prone to invalid log)
  - Penalized logistic regression a.k.a. Logistic ridge regression
  
  Each of them returns the last weight vector and the associated (training) loss.

- `io_helpers.py` : Contains everything related to reading and writing files. In particular .csv files.

- `logistic_regression.py` : Contains functions related to logistic regression for loss, gradient descent and sigmoid.

- `linear_regression.py` : Contains functions related to linear regression : loss and gradient descent.

- `math_helpers.py` : Contains all other math-related functions : polynomial feature expansion, batch_iter for SGD, normalization, prediction, data preprocessing (cf report)

- `cross.py` : Contains all functions related to automatic cross-validation. Our models in total contain 3 hyper-parameter : gamma (learning rate), lambda (penalization for ridge regression) and degree (for polynomial feature expansion). This file offers two functions to select best degree and lambda, as welle as to select best degree, lambda and gamma at the same time. 

- `run.py` : Copy of main.py, containing exact setup that allowed us to reach our highest score on [aicrowd.com](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards). N.B. It contains an alternative version of `logistic_regression` that is resistant to invalid mathematical operations such as 0 in logarithms. It adds an extra variable `epsilon` to avoid undefined values.
