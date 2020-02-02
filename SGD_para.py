#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:39:12 2020

@author: etiennelenaour
"""

#Some imports
import numpy as np
from multiprocessing import Pool


def grad_batch_linreg(batch, x, A, b, lbda):
    
    """Gradient with respect to a sample for ridge loss"""
    
    a_batch = A[batch]
    
    return np.dot(a_batch.T, a_batch.dot(x) - b[batch]) + lbda * x



def sgd_para(x_init, n, n_epochs, size_batch, A, b, lbda):
    
    """Stochastic gradient descent algorithm."""
    
    step = 0.1
    store_every = n
    random_array = np.random.randint(0, n, n * n_epochs * size_batch)
    
    x = x_init.copy()
    x_list = list()
    
    for idx in range(0, n * n_epochs * size_batch, size_batch):
        
        batch = random_array[idx : idx + size_batch]
        x -= step / np.sqrt(idx + 1) * (1 / size_batch) * grad_batch_linreg(batch, x, A, b, lbda)
         
        
        if idx % store_every == 0:
            x_list.append(x.copy())
            
    return x, x_list


def train(parametres):
    A = parametres['A']
    b = parametres['b']
    x_init = parametres['x_init']
    n = parametres['n']
    n_epochs = parametres['n_epochs']
    size_batch = parametres['size_batch']
    lbda = parametres['lbda']
    
    return sgd_para(x_init, n, n_epochs, size_batch, A, b, lbda)   

def sgd_para_run(x_init, n, n_epochs, size_batch, A, b, lbda):
    parametres = [{'A':A[:int(n/2)],'b':b[:int(n/2)],'x_init':x_init,'n':int(n/2 - 1),
                         'n_epochs':n_epochs, 'size_batch':size_batch, 'lbda':lbda},
             {'A':A[int(n/2) + 1:n],'b':b[int(n/2) + 1:n],'x_init':x_init,'n':int(n/2 - 1),
                         'n_epochs':n_epochs, 'size_batch':size_batch, 'lbda':lbda}]
    
    #worker pool
    pool = Pool(2)

    #estimate parameters for each model in parallel
    results = pool.map(train, parametres)
    
    return results
    

    
    
    
