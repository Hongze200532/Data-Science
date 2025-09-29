#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 23:44:22 2023

"""

import numpy as np


def ridge(X, y, l2):
    """ Ridge Regression model with intercept term. (Adopted from https://gist.github.com/wyattowalsh/ea40197ce51b41503bfa188b4ffcecb6)
    L2 penalty and intercept term included via design matrix augmentation.
    This augmentation allows for the OLS estimator to be used for fitting.
    Params:
        X - NumPy matrix, size (N, p), of numerical predictors
        y - NumPy array, length N, of numerical response
        l2 - L2 penalty tuning parameter (positive scalar) 
    Returns:
        NumPy array, length p + 1, of fitted model coefficients
    """
    X = np.expand_dims(X,1) if len(X.shape) == 1 else X # Change shape
    m, n = np.shape(X)
    upper_half = np.hstack((np.ones((m, 1)), X))
    lower = np.zeros((n, n))
    np.fill_diagonal(lower, np.sqrt(l2))
    lower_half = np.hstack((np.zeros((n, 1)), lower))
    X = np.vstack((upper_half, lower_half))
    y = np.append(y, np.zeros(n))
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

def mypolyfit_regularised(x, y, N, L, method='a'):
    """
    Fits polynomial of order N to the (x,y) data, including an L2 norm (ridge) regularisation with paramter lambda.
    Params:
    Returns:
        NumPy array, P are the polynomial coefficients P(1)*X^N + P(2)*X^(N-1) +...+ P(N)*X + P(N+1).
    """
    X = np.zeros((x.shape[0], N+1), dtype=np.float32)
    for i in range(N+1):
        X[:,i] = x**(N-i)

    if method == 'a':
        return ridge(X, y, L)
    if method == 'b':
        '''Ridge regression with sklearn. See link https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html'''
        from sklearn.linear_model import Ridge
        ## Your may have LinAlgWarning. Uncomment to ignore the warning.
        # from scipy.linalg import LinAlgWarning
        # warnings.simplefilter('ignore', LinAlgWarning) 
        clf = Ridge(alpha=L)
        clf.fit(X, y)
        return clf.coef_
    else:
        raise NotImplementedError