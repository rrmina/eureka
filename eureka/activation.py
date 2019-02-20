import numpy as np

# Activation functions
def relu(x):
    x[x<0] = 0
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

def softmax(x):
    numerator = np.exp(x)
    sums = np.sum(numerator, axis=1)
    sums = sums.reshape(sums.shape[0], 1)
    return numerator / sums