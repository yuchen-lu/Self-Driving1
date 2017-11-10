#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 21:21:36 2017

@author: yuchen
"""

'''
gradient descent
sun of the squared errors(SSE)  E=1/2*(sum(y-yhat)
where yhat=f(sum wi*xi)   , f always be sigmoid---prediction
to avoid local minima, method: Momentum
wi = wi+ delta wi, where delta wi = learningrate * -dE/dwi
define error term =(y-yhat)f'(h)
'''
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))


# input data
x = np.array([0.1,0.3])

y = np.array(0.2)


weights = np.array([-0.8, 0.5])

learnrate = 0.5 # eta

# yhat, output of nerual network
nn_output = sigmoid(x[0]*weights[0] + x[1]*weights[1])

# or nn_output = sigmoid(np.dot(x, weights))

error = ( y - nn_output )
error_term = sigmoid_prime(np.dot(x,weights))


# gradient descent step  delta w=learnrate*(y-yhat)*f'(h)*xi
del_w = [learnrate*error*sigmoid_prime(np.dot(weights,x))*x[0], learnrate*error*sigmoid_prime(np.dot(weights,x))*x[1]]

print('nn output:', nn_output)
print('amount of error', error_term)
print('change in weights(delta w)', del_w)