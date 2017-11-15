#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 21:45:24 2017

@author: yuchen
"""

import numpy as np
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


np.random.seed(42)

# hyperparamaters
n_hidden = 2  # number of hidden units, not layers
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape

last_loss = None

# init8ilize weights
weights_input_hidden = np.random.normal(scale = 1/n_features ** .5, size = (n_features, n_hidden))
weights_hidden_output = np.random.normal(scale = 1/n_features ** .5, size = n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    
    for x, y in zip(features.values, targets):
        #forward pass
        hidden_input =  
        hidden_output =
        
        
        
        