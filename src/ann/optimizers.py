"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSprop, Adam, Nadam
All support L2 weight decay.
"""
import numpy as np


class Optimizer:
    """Base optimizer class."""

    def __init__(self, layers, lr=0.001, weight_decay=0.0):
        self.layers = layers
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent."""

    def step(self):
        for layer in self.layers:
            # Gradient already contains L2 term from backward()
            grad_W = layer.grad_W
            grad_b = layer.grad_b
            layer.W -= self.lr * grad_W
            layer.b -= self.lr * grad_b


class Momentum(Optimizer):
    """SGD with Momentum."""

    def __init__(self, layers, lr=0.001, weight_decay=0.0, beta=0.9):
        super().__init__(layers, lr, weight_decay)
        self.beta = beta
        # Initialize velocity for each layer
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            # Gradient already contains L2 term from backward()
            grad_W = layer.grad_W
            grad_b = layer.grad_b

            self.v_W[i] = self.beta * self.v_W[i] + grad_W
            self.v_b[i] = self.beta * self.v_b[i] + grad_b

            layer.W -= self.lr * self.v_W[i]
            layer.b -= self.lr * self.v_b[i]


class NAG(Optimizer):
    """Nesterov Accelerated Gradient.

    Implementation: Uses the look-ahead approach.
    v = beta * v + grad
    W = W - lr * (beta * v + grad)
    """

    def __init__(self, layers, lr=0.001, weight_decay=0.0, beta=0.9):
        super().__init__(layers, lr, weight_decay)
        self.beta = beta
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            # Gradient already contains L2 term from backward()
            grad_W = layer.grad_W
            grad_b = layer.grad_b

            self.v_W[i] = self.beta * self.v_W[i] + grad_W
            self.v_b[i] = self.beta * self.v_b[i] + grad_b

            # Nesterov update: use beta * v_new + grad (look-ahead)
            layer.W -= self.lr * (self.beta * self.v_W[i] + grad_W)
            layer.b -= self.lr * (self.beta * self.v_b[i] + grad_b)


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(self, layers, lr=0.001, weight_decay=0.0, beta=0.9, epsilon=1e-8):
        super().__init__(layers, lr, weight_decay)
        self.beta = beta
        self.epsilon = epsilon
        self.s_W = [np.zeros_like(l.W) for l in layers]
        self.s_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            # Gradient already contains L2 term from backward()
            grad_W = layer.grad_W
            grad_b = layer.grad_b

            self.s_W[i] = self.beta * self.s_W[i] + (1 - self.beta) * grad_W ** 2
            self.s_b[i] = self.beta * self.s_b[i] + (1 - self.beta) * grad_b ** 2

            layer.W -= self.lr * grad_W / (np.sqrt(self.s_W[i]) + self.epsilon)
            layer.b -= self.lr * grad_b / (np.sqrt(self.s_b[i]) + self.epsilon)


class Adam(Optimizer):
    """Adam optimizer with bias correction."""

    def __init__(self, layers, lr=0.001, weight_decay=0.0,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # timestep

        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.m_b = [np.zeros_like(l.b) for l in layers]
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            # Gradient already contains L2 term from backward()
            grad_W = layer.grad_W
            grad_b = layer.grad_b

            # First moment (mean)
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b

            # Second moment (uncentered variance)
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * grad_W ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b ** 2

            # Bias correction
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class Nadam(Optimizer):
    """Nadam optimizer (Adam + Nesterov momentum)."""

    def __init__(self, layers, lr=0.001, weight_decay=0.0,
                 beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(layers, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0

        self.m_W = [np.zeros_like(l.W) for l in layers]
        self.m_b = [np.zeros_like(l.b) for l in layers]
        self.v_W = [np.zeros_like(l.W) for l in layers]
        self.v_b = [np.zeros_like(l.b) for l in layers]

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            # Gradient already contains L2 term from backward()
            grad_W = layer.grad_W
            grad_b = layer.grad_b

            # Update biased first moment
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grad_W
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b

            # Update biased second moment
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * grad_W ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b ** 2

            # Bias-corrected estimates
            m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)

            # Nesterov-corrected first moment:
            # m_nadam = beta1 * m_hat + (1 - beta1) * g / (1 - beta1^t)
            g_W_corrected = grad_W / (1 - self.beta1 ** self.t)
            g_b_corrected = grad_b / (1 - self.beta1 ** self.t)
            m_W_nadam = self.beta1 * m_W_hat + (1 - self.beta1) * g_W_corrected
            m_b_nadam = self.beta1 * m_b_hat + (1 - self.beta1) * g_b_corrected

            layer.W -= self.lr * m_W_nadam / (np.sqrt(v_W_hat) + self.epsilon)
            layer.b -= self.lr * m_b_nadam / (np.sqrt(v_b_hat) + self.epsilon)


def get_optimizer(name, layers, lr=0.001, weight_decay=0.0, **kwargs):
    """
    Factory function to get optimizer by name.

    Args:
        name: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
        layers: list of DenseLayer instances
        lr: learning rate
        weight_decay: L2 regularization coefficient
        **kwargs: additional optimizer-specific parameters

    Returns:
        Optimizer instance
    """
    optimizers = {
        "sgd": SGD,
        "momentum": Momentum,
        "nag": NAG,
        "rmsprop": RMSprop,
        "adam": Adam,
        "nadam": Nadam,
    }
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Choose from {list(optimizers.keys())}")
    return optimizers[name](layers, lr=lr, weight_decay=weight_decay, **kwargs)
