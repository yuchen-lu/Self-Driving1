#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 20:21:33 2017

@author: yuchen
"""

# NOT Perceptron: only care about 1 input, ignore others
weight1=0
weight2=-1.1
bias=1

test_inputs=[(0,0),(0,1),(1,0),(1,1)]
correct_outputs =[True, False, True, False]
outputs = []

# generate and check output
for test_input, correct_output in zip(test_inputs, correct_outputs):
    linear_combin = weight1 * test_input[0] + weight2 * test_input[1] + bias
    output = int(linear_combin >= 0) # 1 if >=0
    is_correct_string = 'Yes' if output == correct_output else 'No'
    outputs.append([test_input[0], test_input[1], linear_combin, output, is_correct_string])


# Print output
num_wrong = len([output[4] for output in outputs if output[4] == 'No'])


#----------highlight: how to print out data nicely using pandas.DataFrame and dict

output_frame = pd.DataFrame(outputs, columns=['Input 1', '  Input 2', '  Linear Combination', '  Activation Output', '  Is Correct'])
if not num_wrong:
    print('Nice!  You got it all correct.\n')
else:
    print('You got {} wrong.  Keep trying!\n'.format(num_wrong))
print(output_frame.to_string(index=False))