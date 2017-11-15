"""
input = X
output of input(1st) layer: weights1*X (sum of wi*xi)
output of hidden layer: f(weights1*x)---y-hat-medium
output of output(3rd) layer: f(weights2*y-hat) ----y-hat-final


columnVector [:,None]

"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:40:45 2017

@author: yuchen
"""

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

# network size
N_input = 4
N_hidden =3
N_output =2
np.random.seed(42)

# make some fake datta

X = np.random.randn(4)

weights_input_to_hidden = np.random.normal(0, scale=0.1, size = (N_hidden,N_input))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size = (N_output,N_hidden))

# make a foraard pass thru the network
hidden_layer_in = np.dot(weights_input_to_hidden,X)
hidden_layer_out = sigmoid(hidden_layer_in)

print('hidden layer out:', hidden_layer_out)

output_layer_in = np.dot(weights_hidden_to_output,hidden_layer_out)
output_layer_out = sigmoid(output_layer_in)

print('output layer out:', output_layer_out)

