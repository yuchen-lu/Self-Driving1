#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
using backprop to find error, then find del(gradient steps)

concepts:
    error@output layer
    error@hidden layer
    gradient steps from hidden to output
    gradient steps from input to hidden
    
At a node, we have a set of h and a , where h =sum of wixi, a is f(h)


'''


"""
Created on Tue Nov 14 21:13:17 2017

@author: yuchen
"""
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

#---------1d
x = np.array([0.5, 0.1, -0.2])
target = 0.6
learnrate = 0.5

weights_input_hidden = np.array([[0.5, -0.6],
                                 [0.1, -0.2],
                                 [0.1, 0.7]])

weights_hidden_output = np.array([0.1, -0.3])

# forward pass

hidden_layer_input = np.dot(x, weights_input_hidden)
hidden_layer_output = sigmoid(hidden_layer_input)


output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
output = sigmoid(output_layer_input)

# bardward pass
# calculate error

error = (target- output)*output*(1-output) 

# calculate error gradient for output layer
del_err_output = error * output *(1-output)

# calcylate error dradient for hidden layer
del_err_hidden = weights_hidden_output * hidden_layer_input * del_err_output

# calculate chage in weights for hidden layer to output layer

del_w_h_o = learnrate * del_err_output * hidden_layer_output

# calculate error gradient for input layer to hidden layer

del_w_i_h = learnrate * del_err_output * x







