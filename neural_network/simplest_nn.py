#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 20:25:28 2017

@author: yuchen
"""

#XOR Perceptron: output 0 if inputs are same, 1 if inputs are different
#logic gate

#passthrough: just passes input to output


import numpy as np


def sigmoid(x):
    result = 1/(1+np.exp(-x))
    return result 



inputs = np.array([0.7, -0.3])
weights = np.array([0.1, 0.8])
bias = -0.1

output = sigmoid(np.dot(weights,inputs)+bias)
print("output is : ",float(output))








