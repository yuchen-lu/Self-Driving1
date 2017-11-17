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
        hidden_input = np.dot(x, weights_input_hidden) #h1,h2
        hidden_output = sigmoid(hidden_input)   #a
        
        output_input = np.dot(hidden_output, weights_hidden_output)
        output = sigmoid(output_input)  # y hat
        
        # backward pass
        # error @ outut layer
        del_err_output = (y-output) * output *(1-output)
    
        #propagate errors in hidden layer
        f_prime_h = hidden_output*(1-hidden_output)
        del_err_hidden=np.dot(del_err_output, f_prime_h)*weights_hidden_output
        
        #  calculate gadient descent steos
        del_w_hidden_output = learnrate* del_err_output*hidden_output   
        '''
# -------------highlight: how to do from size (n,) to size(n,1) i.e. [1,2,3]  to [[1],[2].[3]]   
# -------------- note: all weights matrix not [1 2 3]   but [[1 2] [3 4 ]]
                                              [4 5 6]       [[5 6] [7 8] ]
                                                          
        '''        
        x_bracket = x[:,None]
        del_w_input_hidden = learnrate*  del_err_hidden * x_bracket
        
    #update weights
    weights_input_hidden += learnrate*del_w_input_hidden/n_records
    weights_hidden_output += learnrate*del_w_hidden_output/n_records
    
    
    
    
    # print out the mean square erro on the training set
    if (e%(epochs /10)) ==0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output, weights_hidden_output))
        loss = np.mean((out-targets)**2)
        
        if last_loss and last_loss<loss:
            print("Train loss", loss, "WARNING!--LOSS DEEECREASING")
            
        else:
            print("Train loss", loss)
        last_loss = loss;
            
            
# calculate accuracy on test data
            
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
prediction=out>0.5

accuracy = np.mean(prediction ==targets_test)
print("Prediction accuracy:{:.3f}".format(accuracy))

    


            
        
        
        