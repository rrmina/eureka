import numpy as np

# Dataloader with mini-batch maker
def dataloader(x, y, batch_size=None, shuffle=False):
    """
    Dataloader for making minibatches
    Default is BGD i.e batch_size = All
    """
    # Number of samples
    m = x.shape[0]

    # Shuffle dataset
    if (shuffle):
        permutation = np.random.permutation(m)
        x = x[permutation]
        y = y[permutation]

    # Batch Gradient Descent
    if (batch_size==None):
        return [(x, y)]
    
    # Count the number of minibatches
    num_batches = m // batch_size

    # Make Minibatches
    minibatches = []
    for i in range(num_batches):
        mb_x = x[i*batch_size: (i+1)*batch_size]
        mb_y = y[i*batch_size: (i+1)*batch_size]
        minibatches.append((mb_x, mb_y))

    if (num_batches * batch_size < m):
        mb_x = x[num_batches*batch_size:]
        mb_y = y[num_batches*batch_size:]
        minibatches.append((mb_x, mb_y))

    return minibatches

# One-hot encoder
def one_hot_encoder(y):
    """
    Assumes label of size (batch_size, 1)
    """
    # Get the number of samples
    batch_size = y.shape[0]

    # Reshape (batch_size, 1) to 1-d vector
    y = y.reshape(batch_size)

    # How many zeroes?
    n_values = np.max(y) + 1

    # Initilize zeroes and assign ones
    one_hots = np.zeros((batch_size, n_values))
    one_hots[np.arange(batch_size), y] = 1

    return one_hots