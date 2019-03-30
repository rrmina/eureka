import numpy as np
from .initializer import initialize_weight
from .activation import relu, sigmoid, sigmoid_prime, softmax, tanh, tanh_prime, leaky_relu, elu

# Useful object for stacking of layers
class Sequential(object):
    # Initialization of Sequential Stack
    def __init__(self, layers):
        self.layers = layers
        self.out = None
        self.back_var = None

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        self.out = x
        return self.out

    def backward(self, initial_back_var):
        self.back_var = initial_back_var
        for layer in reversed(self.layers):
            self.back_var = layer.backward(self.back_var)

# Fully-connected Layer
class Linear(object):
    def __init__(self, in_features, out_features, initializer="xavier"):
        # Initialize layer type
        self.layer_type = "nn.Linear"

        # Initialize parameters and gradients
        self.w, self.b = initialize_weight((in_features, out_features), bias=True, initializer=initializer)
        self.dw, self.db = np.zeros((in_features, out_features)), np.zeros((1, out_features))

        # Initialize moments (in case Adam or RMSProp is used as optimizer)
        self.vw, self.vb = np.zeros((in_features, out_features)), np.zeros((1, out_features))
        self.sw, self.sb = np.zeros((in_features, out_features)), np.zeros((1, out_features))
        
        # Storing the input of the layer (aka the actication before the linear layer)
        self.a_prev = None

    def forward(self, x):
        self.a_prev = x
        return np.dot(x, self.w) + self.b

    def backward(self, dh):
        m = dh.shape[0]
        self.dw = (1/m) * np.dot(self.a_prev.T, dh)
        self.db = (1/m) * np.sum(dh, axis=0, keepdims=True)
        return np.dot(dh, self.w.T)



# Activation Functions
class ReLU(object):
    def __init__(self):
        self.layer_type = "activation.ReLU"
        self.relu_prime = None

    def forward(self, x):
        out = relu(x)
        self.relu_prime = out
        return out

    def backward(self, da):
        self.relu_prime[self.relu_prime > 0] = 1
        return da * self.relu_prime

class Sigmoid(object):
    def __init__(self):
        self.layer_type = "activation.Sigmoid"
        self.sigmoid_out = None

    def forward(self, x):
        self.sigmoid_out = sigmoid(x)
        return self.sigmoid_out

    def backward(self, da):
        return da * sigmoid_prime(self.sigmoid_out)

class Softmax(object):
    def __init__(self):
        self.layer_type = "activation.Softmax"
        self.softmax_out = None

    def forward(self, x):
        self.softmax_out = softmax(x)
        return self.softmax_out

    def backward(self, y):
        return self.softmax_out - y

class Tanh(object):
    def __init__(self):
        self.layer_type = "activation.Tanh"
        self.tanh_out = None

    def forward(self, x):
        self.tanh_out = tanh(x)
        return self.tanh_out

    def backward(self, da):
        return da * tanh_prime(self.tanh_out)
    
class LeakyReLU(object):
    def __init__(self):
        self.layer_type = "activation.LeakyReLU"
        self.neg_indices = None
        self.pos_indices = None
        self.leaky_relu_prime = None

    def forward(self, x):
        self.neg_indices = x < 0
        self.pos_indices = x > 0
        self.leaky_relu_prime = x
        return leaky_relu(x)

    def backward(self, da):
        self.leaky_relu_prime[self.pos_indices] = 1
        self.leaky_relu_prime[self.neg_indices] = 0.01
        return da * self.leaky_relu_prime

class ELU(object):
    def __init__(self, alpha=1.0):
        self.layer_type = "activation.ELU"
        self.alpha = np.float32(alpha)
        self.neg_indices = None
        self.pos_indices = None
        self.elu_prime = None

    def forward(self, x):
        self.neg_indices = x < 0
        self.pos_indices = x > 0
        out = elu(x)
        self.elu_prime = out
        return out

    def backward(self, da):
        self.elu_prime[self.pos_indices] = 1
        self.elu_prime[self.neg_indices] += self.alpha
        return da * self.elu_prime 
