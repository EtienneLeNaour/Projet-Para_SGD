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

    """Simulation pour le probleme des moindres carrées"""
    
    d = x.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n_samples)
    noise = std * randn(n_samples)
    b = A.dot(x) + noise
    return A, b


def get_data(d, n_samples):

    """
    utilise simu_linreg pour simuler des données
    d = nombre de variables
    n_sample = taille de l'echantillon
    """
    
    idx = np.arange(d)

    # Ground truth coefficients of the model
    true_coef = (-1)**idx * np.exp(-idx / 10.) # Valeur des vrais paramètres 
    A, b = simu_linreg(true_coef, n_samples)
    
    return A, b, true_coef
