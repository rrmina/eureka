import numpy as np

# Initialize Weights (and biases)
def initialize_weight(shape, bias=True, initializer="xavier"):
    """
    Initialize weights according to initializer
    [1] [ReLU] He Initialization : https://arxiv.org/pdf/1502.01852
    [2] [Tanh] Xavier(Caffe version)
    [3] [Sigmoid] Xavier Initialization : http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    """
    if (initializer=="He"):      # 
        w = np.random.randn(*shape) * np.sqrt(2/shape[0])
        b = np.random.randn(1, shape[1]) * np.sqrt(2/shape[0])
    elif (initializer=="xavier"):  
        w = np.random.randn(*shape) * np.sqrt(1/shape[0])
        b = np.random.randn(1, shape[1]) * np.sqrt(1/shape[0])
    elif (initializer=="xavier_orig"):
        assert shape[0] > shape[1] # Original two-sided Xavier initialization
        w = np.random.randn(*shape) * np.sqrt(2/(shape[0]-shape[1]))
        b = np.random.randn(1, shape[1]) * np.sqrt(2/(shape[0]-shape[1]))
    
    if (~bias):
        b = np.zeros((1, shape[1]))
        
    return w, b