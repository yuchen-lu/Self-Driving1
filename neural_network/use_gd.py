# Mean Square Error: E=1/2m*sum(y-yhat)^2
# used when a lot of data, simply sum leads to big update and diverge gd



#General algorithm:
# 1. set weight step delta_wi =0 ; 
# 2. find output yhat = y-hat = f(sum wixi), error gradi = (y-yhat)*f'(sum wixi)
# 2cont: update weight step delta_wi = delta_wi + delta* xi
# 3. update weights wi = wi + learnrate*delta_wi/m   //average to reduce large variations
# 4. repeat for e epochs


import numpy as np
from data_prep import features, targets, features_test, targets_test


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
# Draw random samples from a normal (Gaussian) distribution
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = .5

for e in range(epochs):
    del_w = np.zeros(weights.shape)
    for x, y in zip(features.values, targets):
        # Loop thru all records, x is the input, y is the target(estimated)


        # output is y-hat  y-hat = f(sum wixi)
        sum_WiXi = np.dot(weights,x)
        
        output = sigmoid(sum_WiXi)

        error_grad = (y-output) * sigmoid_prime(sum_WiXi)

        del_w += error_grad * x

    weights += learnrate * del_w / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss


# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))