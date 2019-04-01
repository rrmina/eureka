import numpy as np
from .initializer import initialize_weight
from .activation import relu, sigmoid, sigmoid_prime, softmax, tanh, tanh_prime, leaky_relu, elu

# Base Class for nn layers/modules
class BaseLayer(object):
    def __init__(self):
        # Affine means that the layer has learnable parameters
        # (i.e weight and bias for Linear/conv | gamma and bias for batch norm)
        self.affine = False

# Useful object for stacking of layers
class Sequential(object):
    # Initialization of Sequential Stack
    def __init__(self, layers):
        self.layers = layers
        self.out = None
        self.back_var = None
        self.train_mode = True

    def forward(self, x):
        for layer in self.layers:
            # Conditional: If test mode, bypass Dropout and Norm layers
            if (self.train_mode):
                x = layer.forward(x)
            else:
                if (layer.layer_type != "nn.Dropout"):
                    x = layer.forward(x)
        self.out = x
        return self.out

    def backward(self, initial_back_var):
        self.back_var = initial_back_var
        for layer in reversed(self.layers):
            self.back_var = layer.backward(self.back_var)

    def train(self):
        self.train_mode = True
    
    def test(self):
        self.train_mode = False

# Fully-connected Layer
class Linear(BaseLayer):
    def __init__(self, in_features, out_features, initializer="xavier"):
        super(Linear, self).__init__()
        self.affine = True

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

# Dropout Layer
class Dropout(BaseLayer):
    def __init__(self, drop_prob):
        super(Dropout, self).__init__()
        # Initialize layer type
        self.layer_type = "nn.Dropout"

        # Initialize parameters
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

        # Initialize mask
        self.mask = None

    def forward(self, x):
        # Generate mask
        self.mask = np.random.binomial(1, self.keep_prob, size=x.shape)

        # Apply mask
        return x * self.mask

    def backward(self, da):
        return da * self.mask

# Normalization Layers
class BatchNorm1d(BaseLayer):
    """
    Reference: https://kevinzakka.github.io/2016/09/14/batch_normalization/
    To be consistent with the naming convention of layers with affine parameters (e.g. nn.Linear), 
    we rename gamma and beta as w and b
    This also reduces unnecessary codes implementing gradient descent in optim due to different naming convention 
    """
    def __init__(self, num_features, epsilon=1e-8, affine=False):
        super(BatchNorm1d, self).__init__()
        self.affine = affine

        # Layer type
        self.layer_type = "nn.BatchNorm1d"

        # Hyperparameters
        self.epsilon = epsilon

        # Class variables
        self.x_hat = None
        self.u = None
        self.std = None
        self.batch_size = None
        
        # Affine (Learnable) parameters
        self.w = np.ones((1, num_features))     # gamma
        self.b = np.zeros((1, num_features))    # beta

        # For the most part of BatchNorm, you don't actually need to implement these gradient variables
        if (self.affine):
            self.dw = np.zeros((1, num_features)), np.zeros((1, num_features))
            self.vw, self.vb = np.zeros((1, num_features)), np.zeros((1, num_features))
            self.sw, self.sb = np.zeros((1, num_features)), np.zeros((1, num_features))

    def forward(self, x):
        # Class variables
        self.m = x.shape[0] # batch size

        # Mean per feature over minibatch
        self.u = np.mean(x, axis=0)

        # Standard Deviation per feature over minibatch
        self.std = np.sqrt(np.var(x, axis = 0) + self.epsilon)

        # Normalize
        self.x_hat = (x - self.u)/self.std

        # Scale and Shift
        out = self.x_hat * self.w + self.b

        return out

    def backward(self, d_bn_out):
        # Gradient with respect to affine parameters
        if (self.affine):
            self.dbeta = np.sum(d_bn_out, axis=0)
            self.dgamma = np.sum(d_bn_out*self.x_hat, axis=0)

        # Gradient of loss with respect to BN-layer input x
        dx_hat = d_bn_out * self.w
        numerator = self.m * dx_hat - np.sum(dx_hat, axis=0) - self.x_hat*np.sum(dx_hat*self.x_hat, axis=0)
        dx = numerator/(self.m * self.std)

        return dx

# Activation Functions
class ReLU(BaseLayer):
    def __init__(self):
        super(ReLU, self).__init__()
        self.layer_type = "activation.ReLU"
        self.relu_prime = None

    def forward(self, x):
        out = relu(x)
        self.relu_prime = out
        return out

    def backward(self, da):
        self.relu_prime[self.relu_prime > 0] = 1
        return da * self.relu_prime

class Sigmoid(BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.layer_type = "activation.Sigmoid"
        self.sigmoid_out = None

    def forward(self, x):
        self.sigmoid_out = sigmoid(x)
        return self.sigmoid_out

    def backward(self, da):
        return da * sigmoid_prime(self.sigmoid_out)

class Softmax(BaseLayer):
    def __init__(self):
        super(Softmax, self).__init__()
        self.layer_type = "activation.Softmax"
        self.softmax_out = None

    def forward(self, x):
        self.softmax_out = softmax(x)
        return self.softmax_out

    def backward(self, y):
        return self.softmax_out - y

class Tanh(BaseLayer):
    def __init__(self):
        super(Tanh, self).__init__()
        self.layer_type = "activation.Tanh"
        self.tanh_out = None

    def forward(self, x):
        self.tanh_out = tanh(x)
        return self.tanh_out

    def backward(self, da):
        return da * tanh_prime(self.tanh_out)
    
class LeakyReLU(BaseLayer):
    def __init__(self):
        super(LeakyReLU, self).__init__()
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

class ELU(BaseLayer):
    def __init__(self, alpha=1.0):
        super(ELU, self).__init__()
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
