import os
import pickle
import tarfile
import numpy as np
from urllib.request import urlretrieve

"""
Reference: https://mattpetersen.github.io/load-cifar10-with-numpy
"""

url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
tar = 'cifar-10-binary.tar.gz'
path = "data/"
train_files = [
    'cifar-10-batches-bin/data_batch_1.bin',
    'cifar-10-batches-bin/data_batch_2.bin',
    'cifar-10-batches-bin/data_batch_3.bin',
    'cifar-10-batches-bin/data_batch_4.bin',
    'cifar-10-batches-bin/data_batch_5.bin'
]

test_files = [
    'cifar-10-batches-bin/test_batch.bin'
]

def load_dataset(download=True, train=True):
    # Check if data/ directory exists
    if not os.path.exists(path):
        os.makedirs(path)

    # Download tarfile if missing
    # Actually, the download arg isn't used here
    file_path = os.path.join(path, tar)
    if not os.path.exists(file_path):
        print("Downloading ", tar, " ... ")
        urlretrieve(url, file_path)

    # Load data from tarfile
    with tarfile.open(os.path.join(path, tar)) as tar_object:
        # Each file contains 10,000 color images and 10,000 labels
        fsize = 10000 * (32 * 32 * 3) + 10000

        if (train):
            # There are 6 files (5 train and 1 test)
            buffr = np.zeros(fsize * 5, dtype='uint8')

            # Get members of tar corresponding to data files
            # -- The tar contains README's and other extraneous stuff
            members = [file for file in tar_object if file.name in train_files]

            # Sort those members by name
            # -- Ensures we load train data in the proper order
            # -- Ensures that test data is the last file in the list
            members.sort(key=lambda member: member.name)

        else:
            buffr = np.zeros(fsize * 1, dtype='uint8')
            members = [file for file in tar_object if file.name in test_files]

        # Extract data from members
        for i, member in enumerate(members):
            # Get member as a file object
            f = tar_object.extractfile(member)
            # Read bytes from that file object into buffr
            buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

    # Parse data from buffer
    # -- Examples are in chunks of 3,073 bytes
    # -- First byte of each chunk is the label
    # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

    # Labels are the first byte of every chunk
    labels = buffr[::3073]

    # Pixels are everything remaining after we delete the labels
    pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
    images = pixels.reshape(-1, 3072).astype('float32') / 255

    return images.reshape(-1, 3, 32, 32), labels.reshape(-1, 1)
            