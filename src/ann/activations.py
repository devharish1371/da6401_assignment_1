"""
Activation Functions and Their Derivatives
Implements: ReLU, Sigmoid, Tanh, Softmax
"""
import numpy as np


class Sigmoid:
    """Sigmoid activation: σ(x) = 1 / (1 + e^(-x))"""

    def forward(self, x):
        # Clip to avoid overflow in exp
        x = np.clip(x, -500, 500)
        self.output = 1.0 / (1.0 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        """dL/dz = dL/dy * y * (1 - y)"""
        return grad_output * self.output * (1.0 - self.output)


class Tanh:
    """Tanh activation: tanh(x)"""

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        """dL/dz = dL/dy * (1 - y^2)"""
        return grad_output * (1.0 - self.output ** 2)


class ReLU:
    """ReLU activation: max(0, x)"""

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        """dL/dz = dL/dy * indicator(x > 0)"""
        return grad_output * (self.input > 0).astype(np.float64)


class Softmax:
    """
    Softmax activation: e^(x_i) / sum(e^(x_j))
    Numerically stable implementation using max subtraction.
    """

    def forward(self, x):
        # Subtract max for numerical stability
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        """
        Compute Jacobian product for Softmax.
        dL/dz_i = p_i * (dL/dy_i - sum_j(dL/dy_j * p_j))
        
        This allows Softmax to be used with any loss function.
        - For CrossEntropy: dL/dy = -y/p. Result: p - y.
        - For MSE: dL/dy = p - y. Result: p*(p-y - sum(p*(p-y))).
        """
        # grad_output is dL/dy (batch_size, num_classes)
        # self.output is p (batch_size, num_classes)
        
        sum_term = np.sum(grad_output * self.output, axis=1, keepdims=True)
        return self.output * (grad_output - sum_term)


def get_activation(name):
    """
    Factory function to get activation by name.

    Args:
        name: 'sigmoid', 'tanh', 'relu', or 'softmax'

    Returns:
        Activation class instance
    """
    activations = {
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "relu": ReLU,
        "softmax": Softmax,
    }
    name = name.lower()
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
    return activations[name]()
