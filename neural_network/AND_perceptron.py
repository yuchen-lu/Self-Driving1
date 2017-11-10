# -*- coding: utf-8 -*-
"""
Spyder Editor

"""

#pandas.DataFrame
# activation function by combining linear conbin

import pandas as pd

# for AND perceptron, need both output be 1
# output =1 if w1*in1+w2*in2+bias>=0


weight1=1.1
weight2=1.1
bias=-2

test_inputs=[(0,0),(0,1),(1,0),(1,1)]
correct_outputs =[False, False, False, True]
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
