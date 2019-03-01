import numpy as np

# Fast PCA implementation using np.linalg.eigh()
# Faster by around 3 times in small datasets
def PCA(data, num_dim):
    # Reshape the data into flattened vectors [num_samples, -1]
    original_shape = data.shape
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
    
    # Print PCA Stats
    diff = data - data_recovered
    error = np.sum(diff*diff) / np.sum(data*data)
    print("PCA with {} Principal Components - {}% variance retained".format(num_dim, (1-error)*100))

    # Add mean
    data_recovered += mean

    # Reshape to original shape
    data_recovered = data_recovered.reshape(*original_shape)

    return data_recovered, V, S

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
    
    # Print PCA Stats
    diff = data - data_recovered
    error = np.sum(diff*diff) / np.sum(data*data)
    print("PCA with {} Principal Components - {}% variance retained".format(num_dim, (1-error)*100))

    # Add mean
    data_recovered += mean

    # Reshape to original shape
    data_recovered = data_recovered.reshape(*original_shape)

    return data_recovered, U, S

# Search for the least Principal Comp. satisfying variance condition
def PCA_search(data, U, variance=0.01):
    # Reshape the data into flattened vectors [num_samples, -1]
    original_shape = data.shape
    num_samples = data.shape[0]
    data = data.reshape(num_samples, -1) 
    
    # Get the mean and zero center the data
    data_original = data
    mean = np.mean(data, axis=0) # per pixel mean
    data = data - mean    
    
    # Get the denominator
    denominator = np.sum(data_original*data_original)
    right_side = variance * denominator
    
    # Number of Principal Components
    num_pcomp = data.shape[1]
    
    # Find the least number of Principal Components needed to satisfy variance condition
    # Binary Search
    upperbound = data.shape[1]
    lowerbound = 0
    i = upperbound//2
    last_i = -1
    while (1):
        # Usual PCA Compression and Recovery
        data_compressed = np.dot(data, U[:, :i])
        data_recovered = np.dot(data_compressed, U[:, :i].transpose())
        data_recovered += mean
        
        # Squared Error
        diff = data_original - data_recovered
        numerator = np.sum(diff*diff)
        left_side = numerator
        
        if (left_side > right_side):
            lowerbound = i
            i += (upperbound-lowerbound)//2
        else:
            upperbound = i
            i -= (upperbound-lowerbound)//2

        if (last_i == i):
            break
            
        last_i = i
    
    # Print PCA Stats
    diff = data_original - data_recovered
    error = np.sum(diff*diff) / denominator
    print("PCA with {} Principal Components - {}% variance retained".format(i, (1-error)*100))
    
    # Reshape recovered data
    data_recovered = data_recovered.reshape(*original_shape)
    
    return data_recovered, i