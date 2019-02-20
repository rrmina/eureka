import numpy as np

# Activation functions
def relu(x):
    x[x<0] = 0
    return x

def leaky_relu(x):
    x[x<0] *= np.float32(0.01)
    return x

def elu(x, alpha=1.0):
    x[x<0] = np.float32(alpha) * (np.exp(x[x<0]) - 1)
    return x

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

def tanh(x):
    #return 2 * sigmoid(2*x) - 1
    return 2/(1+np.exp(-2*x)) -1

def tanh_prime(x):
    return 1-x*x

def softmax(x):
    numerator = np.exp(x)
    sums = np.sum(numerator, axis=1)
    sums = sums.reshape(sums.shape[0], 1)
    return numerator / sums