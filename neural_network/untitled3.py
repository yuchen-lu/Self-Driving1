#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
topological.sort() returns a sorted list of nodes, all calc can run in series
topological.sort() takes in a feed_dict ---> how we initialize value in Input node

eg for feed_dict:

# define 2 Input nodes
x y = Input.(), Input.()

# define 1 'Add' node, two Input nodes being the input
add = Add (x,y)

# valye of 'x' 'y' set to 10&20
feed_dict ={x:10, y:20}

#sort the nodes with topoligical sort
sorted_nodes =topoligical_sort(feed_dict = feed_dict)

'''
"""
Created on Thu Nov 16 20:07:32 2017

@author: yuchen
"""

class node(object):
    def _init_(self, inbound_nodes=[]):
    
        # Node(s) from which this node receives values
        self.inbound_nodes = inbound_nodes
        # nodes to which this node passes values
        self.outbound_nodes =[]
        # for each inbound node, add this node as an outbound node to _that_ node
        
        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)
            
        self.value = None
        
    def forward(self):
        '''
        forward propgation
        compute output value based on 'inbound_nodes' and store the result in self.value
        '''
        
        raise NotImplemented
        
class input(Node):
    
    '''
    this class(subclass of Node) not compute
    only store values, such as data feature or model params
    set value explicitly or with forward()
    '''
    def _init_(self):
        
        # input node is the only nodewjere value passed as forward().
        # all other node get value from prev nodes by self.inbound_nodes
        # eg: val0 = self.inbound_nodes[0].value
        
    def forward(self, value = None):
        if value is not None:
            self.value = value
            
            
class Add(Node):
    def _init_(self, [x,y]):
        node._init_(self, [x,y])
        
    def forward(self):
        