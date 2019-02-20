import numpy as np
from .initializer import initialize_weight
from .activation import relu, sigmoid, sigmoid_prime, softmax

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

    def backward(self, label):
        for layer in reversed(self.layers):
            # Assuming Softmax and Cross-entropy loss
            if (layer.layer_type == "activation.Softmax"):
                self.back_var = layer.backward(label)
            else:
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

    def forward(self, x):
        return relu(x)

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