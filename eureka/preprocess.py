import numpy as np

# Fast PCA implementation using np.linalg.eigh()
# Faster by around 3 times in small datasets
def PCA(data, num_dim):
    # Reshape the data into flattened vectors [num_samples, -1]
    num_samples = data.shape[0]
    data = data.reshape(num_samples, -1)

    # Get the mean and zero center the data
    mean = np.mean(data, axis=0) # per pixel mean
    data = data - mean

    # Get the covariance matrix, Numpy has few tricks in computing covariance
    cov = np.cov(data.T)
    
    # Perform factor Decomposition, using SVD
    S, V = np.linalg.eigh(cov)
    
    # V is originally sorted in ascending eigenvalue, we want the opposite sorting
    V = np.fliplr(V)
    
    # Project the data using fewer eigenvectors - [Compressed Representation] 
    data_projected = np.dot(data, V[:, :num_dim])
    
    # Recover data from the compressed representation
    data_recovered = np.dot(data_projected, V[:, :num_dim].transpose())
    
    # Add mean
    data_recovered += mean
    
    return data_recovered, V

# Performs PCA using SVD
def PCA_svd(data, num_dim):
    # Reshape the data into flattened vectors [num_samples, -1]
    num_samples = data.shape[0]
    data = data.reshape(num_samples, -1)

    # Get the mean and zero center the data
    mean = np.mean(data, axis=0) # per pixel mean
    data = data - mean

    # Get the covariance matrix
    cov = np.dot(data.T, data)
    
    # Perform factor Decomposition, using SVD
    U, S, V = np.linalg.svd(cov)
    
    # Project the data using fewer eigenvectors - [Compressed Representation] 
    data_compressed = np.dot(data, U[:, :num_dim])
    
    # Recover data from the compressed representation
    data_recovered = np.dot(data_compressed, U[:, :num_dim].transpose())
    
    # Add mean
    data_recovered += mean
    
    return data_recovered, U