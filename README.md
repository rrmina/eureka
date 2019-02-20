# Eureka : A simple Neural Network Framework written in Numpy :zap: :bulb: :high_brightness:

### Clean Interface

#### Loading Datasets in-house

```python
import datasets.mnist

train_x, train_y = datasets.mnist.load_dataset(download=True, train=True)
test_x, test_y = datasets.mnist.load_dataset(download=True, train=False)
```

#### Dataloader and Minibatch Maker

```python
from utils import dataloader

trainloader = dataloader(x, y, batch_size=64, shuffle=True)
```

#### Defining Model Architecture and Optimizer

```python
import eureka.nn as nn
import eureka.optim as optim

# MNIST Dense network with 1-hidden layer of 256 neurons
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.Sigmoid(),
    nn.Linear(256, 10),
    nn.Softmax()
])

# Adam Optimizer
optimizer = optim.Adam(model, lr=0.0002)
```

#### Forward and Backpropagation

```python
for inputs, labels in trainloader:
    # Forward Propagation and Compute loss
    out = model.forward(inputs)
    m = inputs.shape[0]
    batch_loss += cross_entropy_loss(out, labels.argmax(axis=1).reshape(m,1))

    # Compute Gradients and Backward Prop using Optimizer step
    model.backward(labels)
    optimizer.step()
```

## Example: MNIST Classification

```python
import numpy as np
from eureka.utils import one_hot_encoder, dataloader
from eureka.losses import cross_entropy_loss
import eureka.optim as optim
import eureka.nn as nn
import datasets.mnist

# Load dataset and Preprocess
train_x, train_y = datasets.mnist.load_dataset(download=True, train=True)
x = train_x.reshape(train_x.shape[0], -1)
y = one_hot_encoder(train_y)
num_samples = x.shape[0]

# Prepare the dataloader
trainloader = dataloader(x, y, batch_size=64, shuffle=True)

# Define model architecture and Optimizer
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.Sigmoid(),
    nn.Linear(256, 10),
    nn.Softmax()
])
optimizer = optim.Adam(model, lr=0.0002)

# Train loop
num_epochs = 20
for epoch in range(1, num_epochs+1):
    print("Epoch: {}/{}\n==========".format(epoch, num_epochs))
    acc = 0
    batch_loss = 0
    for inputs, labels in trainloader:
        # Number of samples per batch
        m = inputs.shape[0]

        # Forward Propagation and Compute loss
        out = model.forward(inputs)
        batch_loss += cross_entropy_loss(out, labels.argmax(axis=1).reshape(m,1))

        # Compute Gradients and Backward Prop using Optimizer step
        model.backward(labels)
        optimizer.step()
    
    # Print Loss
    print("Loss: {:.6f}".format(batch_loss/num_samples))
        
```