import numpy as np

class BaseLoss(object):
    def __init__(self):
        self.loss = 0
        self.back_var = None

    def forward(self):
        pass

    def __call__(self, outputs, targets):
        return self.forward(outputs, targets)

    def backward(self):
        pass 

class CrossEntropyLoss(BaseLoss):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        # For backprop - Efficient Softmax-CrossEntropy Combo
        self.back_var = targets

        # Efficient negative log likelihood
        batch_size = targets.shape[0]
        targets = targets.argmax(axis=1).reshape(batch_size, 1)
        targets = targets.reshape(batch_size)

        # [Negative log-likelihood] Only get the negative logs of one-hot targets
        log_likelihood = -np.log(outputs[np.arange(batch_size), targets])
        self.loss = np.sum(log_likelihood)

        return self.loss

    def backward(self):
        return self.back_var

class MSELoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.outputs = None
        self.targets = None

    def forward(self, outputs, targets):
        self.outputs = outputs
        self.targets = targets
        return 0.5 * np.mean(np.power(outputs - targets, 2))

    def backward(self):
        return self.outputs - self.targets
        