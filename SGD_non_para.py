#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 12:34:03 2020

@author: etiennelenaour
"""

import numpy as np


def grad_batch_linreg(batch, x, A, b, lbda, n_samples):
    
    """Gradient with respect to a sample for ridge loss"""
    
    a_batch = A[batch]
    
    return np.dot(a_batch.T, a_batch.dot(x) - b[batch]) + lbda * x



def sgd(x_init, n, n_epochs, size_batch, A, b, lbda, step=0.1):
    
    """Stochastic gradient descent algorithm."""
    
    store_every = n
    random_array = np.random.randint(0, n, n * n_epochs * size_batch)
    
    x = x_init.copy()
    x_list = list()
    
    for idx in range(0, n * n_epochs * size_batch, size_batch):
        
        batch = random_array[idx : idx + size_batch]
        x -= step / np.sqrt(idx + 1) * (1 / size_batch) * grad_batch_linreg(batch, x, A, b, lbda, n)
         
        
        if idx % store_every == 0:
            x_list.append(x.copy())
            
    return x, x_list
