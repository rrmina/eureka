import numpy as np

# Optimizers
class SGD(object):
    def __init__(self, model_instance, lr=0.001):
        self.model_instance = model_instance
        self.lr = lr

    def step(self):
        for layer in self.model_instance.layers:
            if (layer.affine):
                # Weight Update
                layer.w -= self.lr * layer.dw
                layer.b -= self.lr * layer.db

class Momentum(object):
    def __init__(self, model_instance, lr=0.01, beta_1=0.9):
        self.model_instance = model_instance
        self.lr = lr
        self.beta_1 = beta_1

    def step(self):
        for layer in self.model_instance.layers:
            if (layer.affine):
                # Compute 1st moment
                layer.vw = self.beta_1 * layer.vw + (1-self.beta_1) * layer.dw
                layer.vb = self.beta_1 * layer.vb + (1-self.beta_1) * layer.db

                # Weight Update
                layer.w -= self.lr * layer.vw
                layer.b -= self.lr * layer.vb

class RMSProp(object):
    def __init__(self, model_instance, lr=0.0001, beta_2=0.999, epsilon=1e-8):
        self.model_instance = model_instance
        self.lr = lr
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def step(self):
        for layer in self.model_instance.layers:
            if (layer.affine):
                # Compute 2nd moment
                layer.sw = self.beta_2 * layer.sw + (1-self.beta_2) * layer.dw * layer.dw
                layer.sb = self.beta_2 * layer.sb + (1-self.beta_2) * layer.db * layer.db

                # Weight Update
                layer.w -= self.lr * layer.dw / (np.sqrt(layer.sw) + self.epsilon)
                layer.b -= self.lr * layer.db / (np.sqrt(layer.sb) + self.epsilon)

class Adam(object):
    def __init__(self, model_instance, lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.model_instance = model_instance
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def step(self):
        for layer in self.model_instance.layers:
            if (layer.affine):
                # Compute 1st moment
                layer.vw = self.beta_1 * layer.vw + (1-self.beta_1) * layer.dw
                layer.vb = self.beta_1 * layer.vb + (1-self.beta_1) * layer.db

                # Compute 2nd moment
                layer.sw = self.beta_2 * layer.sw + (1-self.beta_2) * layer.dw * layer.dw
                layer.sb = self.beta_2 * layer.sb + (1-self.beta_2) * layer.db * layer.db

                # Weight Update
                layer.w -= self.lr * layer.vw / (np.sqrt(layer.sw) + self.epsilon)
                layer.b -= self.lr * layer.vb / (np.sqrt(layer.sb) + self.epsilon)

class Adam_accident(object):
    def __init__(self, model_instance, lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.model_instance = model_instance
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

    def step(self):
        for layer in self.model_instance.layers:
            if (layer.affine):
                
                # Compute 1st moment
                # This is a wrong mistake! See what I did there? negative * negative = positive?
                # It turns out that this 1st moment computation produces empirically good results WOW Eureka!
                # Sometimes, it performs better than the vanilla Adam
                # Becase the 1st moment is "pre-multiplied" by a small number, we can now use larger learning rates,
                # expanding the range of "effective" learning rate values for Adam 
                layer.vw = (self.beta_1 * layer.vw + (1-self.beta_1) * layer.dw) * (1 - self.beta_1)
                layer.vb = (self.beta_1 * layer.vb + (1-self.beta_1) * layer.db) * (1 - self.beta_1)

                # Compute 2nd moment
                layer.sw = self.beta_2 * layer.sw + (1-self.beta_2) * layer.dw * layer.dw
                layer.sb = self.beta_2 * layer.sb + (1-self.beta_2) * layer.db * layer.db

                # Weight Update
                layer.w -= self.lr * layer.vw / (np.sqrt(layer.sw) + self.epsilon)
                layer.b -= self.lr * layer.vb / (np.sqrt(layer.sb) + self.epsilon)


# Learning Rate Decay and Schedulers

