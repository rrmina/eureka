import numpy as np
import struct
import os
import urllib.request as ur
import gzip
from PIL import Image
import codecs

# External file info
mnist_url = "http://yann.lecun.com/exdb/mnist/"
mnist_download_folder = "data/"

# Filenames
mnist_filenames = {   
    "test_image" : "t10k-images-idx3-ubyte", 
    "test_label": "t10k-labels-idx1-ubyte", 
    "train_image" : "train-images-idx3-ubyte", 
    "train_label": "train-labels-idx1-ubyte" 
}
file_bytes = {   
    "test_image" : 1648877, 
    "test_label": 4542, 
    "train_image" : 9912422, 
    "train_label": 28881 
}

def download_one( filename, expected_bytes, debug=0, gz=0 ):
    """
    Download a file if not present, and make sure it's the right size.
    Files are stored in \'data\' folder
    """
    filename = filename + ".gz"
    filepath = mnist_download_folder + filename

    if not os.path.exists( mnist_download_folder ):
        os.makedirs( mnist_download_folder )

    if not os.path.exists( filepath ):
        print( "Downloading ", filename, " ..." )
        file_download = ur.URLopener()
        file_download.retrieve( mnist_url + filename, filepath )
        statinfo = os.stat( filepath )
        if statinfo.st_size == expected_bytes:
            if (debug):
                print( "Found and verified", filepath )
        else:
            raise Exception( "Failed to verify " +
                            filename + ". Can you get to it with a browser? \nDownload .gz files from http://yann.lecun.com/exdb/mnist/ and store in mnist_download folder" )
    else:
        print( "Found and verified", filepath )

    return filepath

def load_dataset(download=True, train=True):
    """
    Downloads gzip file and returns image data and label
    """
    if (download):
        mnist_download()
    if (train):
        data = read_image_file("data/" + mnist_filenames["train_image"]+".gz")
        label = read_label_file("data/" + mnist_filenames["train_label"]+".gz")
    else:
        data = read_image_file("data/" + mnist_filenames["test_image"]+".gz")
        label = read_label_file("data/" + mnist_filenames["test_label"]+".gz")

    return data, label

def mnist_download(debug=0):
    for key in mnist_filenames:
        if (debug):
            print(download_one(mnist_filenames[key], file_bytes[key]))
        else:
            download_one(mnist_filenames[key], file_bytes[key])

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)

def read_label_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        
    return parsed.reshape(length, -1)

def read_image_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        n_rows = get_int(data[8:12])
        n_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
    
    return parsed.reshape(length, n_rows, n_cols) / 255
