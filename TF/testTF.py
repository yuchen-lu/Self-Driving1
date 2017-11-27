#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 18:01:15 2017

@author: yuchen
"""

import tensorflow as tf
hello_constant = tf.constant('Hello World')

with tf.Session() as sess:
    output = sess.run(hello_constant)
    print(output)