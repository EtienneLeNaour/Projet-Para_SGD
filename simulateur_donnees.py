#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 13:42:15 2020

@author: etiennelenaour
"""
import numpy as np

from numpy.random import multivariate_normal, randn
from scipy.linalg.special_matrices import toeplitz


def simu_linreg(x, n_samples, std=1., corr=0.5):
    """Simulation for the least-squares problem.

    Parameters
    ----------
    x : ndarray, shape (d,)
        The coefficients of the model
    n : int
        Sample size
    std : float, default=1.
        Standard-deviation of the noise
    corr : float, default=0.5
        Correlation of the features matrix
    
    Returns
    -------
    A : ndarray, shape (n, d)
        The design matrix.
    b : ndarray, shape (n,)
        The targets.
    """
    d = x.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n_samples)
    noise = std * randn(n_samples)
    b = A.dot(x) + noise
    return A, b


def get_data(d, n_samples):

    """
    use simu_linreg in order to simulate data
    d = nb features
    n_sample = nb samples
    """
    idx = np.arange(d)

    # Ground truth coefficients of the model
    x_model_truth = (-1)**idx * np.exp(-idx / 10.) # Valeur des vrais paramètres 
    A, b = simu_linreg(x_model_truth, n_samples)
    
    return A, b, x_model_truth