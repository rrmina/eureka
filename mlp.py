
import numpy as np
from eureka.activation import relu, sigmoid, sigmoid_prime, softmax
from eureka.utils import one_hot_encoder, dataloader
from eureka.losses import cross_entropy_loss
import eureka.optim as optim
import eureka.nn as nn
import datasets.mnist

def main():
    # Load dataset  
    train_x, train_y = datasets.mnist.load_dataset(download=True, train=True)

    # Preprocess Dataset
    x = train_x.reshape(train_x.shape[0], -1)
    y = one_hot_encoder(train_y)
    num_samples = x.shape[0]

    # Prepare the dataloader
    trainloader = dataloader(x, y, batch_size=64, shuffle=True)

    # Define model's architecture
    model = nn.Sequential([
        nn.Linear(784, 256),
        nn.Sigmoid(),
        nn.Linear(256, 10),
        nn.Softmax()
    ])

    # Define the optimizer
    optimizer = optim.Adam(model, lr=0.0002)

    num_epochs = 20
    for epoch in range(1, num_epochs+1):
        print("Epoch: {}/{}\n==========".format(epoch, num_epochs))
        acc = 0
        batch_loss = 0
        for inputs, labels in trainloader:
            # Number of samples per batch
            m = inputs.shape[0]

            # Forward Propagation
            out = model.forward(inputs)

            # Compute accuracy
            pred = np.argmax(out, axis=1)
            pred = pred.reshape(pred.shape[0], 1)
            acc += np.sum(pred == labels.argmax(axis=1).reshape(m,1))
            
            # Compute loss
            batch_loss += cross_entropy_loss(out, labels.argmax(axis=1).reshape(m,1))

            # Backward Propagation
            model.backward(labels)

            # Optimization Step
            optimizer.step()
        
        print("Loss: {:.6f}".format(batch_loss/num_samples))
        print("Accuracy: {:.2f}%\n".format(acc/num_samples*100))
            
main()