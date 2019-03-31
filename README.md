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

#### Defining Model Architecture, Optimizer, and Criterion/Loss Function

```python
import eureka.nn as nn
import eureka.optim as optim
import eureka.losses as losses

# MNIST Dense network with 1-hidden layer of 256 neurons and a Dropout of 0.5
model = nn.Sequential([
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
    nn.Softmax()
])

# Adam Optimizer
optimizer = optim.Adam(model, lr=0.0002)

# Define the criterion/loss function
criterion = losses.CrossEntropyLoss()
```

#### Forward and Backpropagation

```python
for inputs, labels in trainloader:
    # Forward Propagation and Compute loss
    out = model.forward(inputs)
    m = inputs.shape[0]
    batch_loss += criterion(out, labels)

    # Compute Loss and Model Gradients
    back_var = criterion.backward()
    model.backward(labels)

    # Backward Prop using Optimizer step
    optimizer.step()
```

## Example: MNIST Classification

```python
import numpy as np
from eureka.utils import one_hot_encoder, dataloader
import eureka.losses as losses
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
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10),
    nn.Softmax()
])
optimizer = optim.Adam(model, lr=0.0002)
criterion = losses.CrossEntropyLoss()

# Train loop
num_epochs = 20
for epoch in range(1, num_epochs+1):
    print("Epoch: {}/{}\n==========".format(epoch, num_epochs))
    acc = 0
    batch_loss = 0
    for inputs, labels in trainloader:
        # Forward Propagation and Compute loss
        out = model.forward(inputs)
        m = inputs.shape[0]
        batch_loss += criterion(out, labels)

        # Compute Accuracy
        pred = np.argmax(out, axis=1).reshape(-1, 1)
        acc += np.sum(pred == labels.argmax(axis=1).reshape(-1,1))
        
        # Compute Loss and Model Gradients
        back_var = criterion.backward()
        model.backward(labels)

        # Backward Prop using Optimizer step
        optimizer.step()
    
    # Print Loss and Accuracy
    print("Loss: {:.6f}".format(batch_loss/num_samples)) 
    print("Accuracy: {:.2f}%\n".format(acc/num_samples*100)) 
```