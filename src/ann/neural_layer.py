"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""
import numpy as np


class DenseLayer:
    """
    Fully connected (dense) layer.

    Stores self.grad_W and self.grad_b after every backward() call
    for gradient verification by the autograder.
    """

    def __init__(self, input_size, output_size, weight_init="xavier"):
        """
        Initialize layer weights and biases.

        Args:
            input_size: number of input features
            output_size: number of output features (neurons)
            weight_init: 'random', 'xavier', or 'zeros'
        """
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights
        if weight_init == "xavier":
            # Xavier/Glorot initialization
            std = np.sqrt(2.0 / (input_size + output_size))
            self.W = np.random.randn(input_size, output_size) * std
        elif weight_init == "random":
            self.W = np.random.randn(input_size, output_size) * 0.01
        elif weight_init == "zeros":
            self.W = np.zeros((input_size, output_size))
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")

        # Biases always initialized to zeros
        self.b = np.zeros((1, output_size))

        # Gradient placeholders (set during backward)
        self.grad_W = None
        self.grad_b = None

        # Cached input for backward pass
        self.input = None

    def forward(self, X):
        """
        Forward pass: output = X @ W + b

        Args:
            X: input of shape (batch_size, input_size)

        Returns:
            Output of shape (batch_size, output_size)
        """
        self.input = X
        return X @ self.W + self.b

    def backward(self, grad_output):
        """
        Backward pass: compute gradients and return gradient for previous layer.

        Args:
            grad_output: gradient from next layer, shape (batch_size, output_size)

        Returns:
            Gradient for previous layer, shape (batch_size, input_size)
        """
        # Gradient w.r.t. weights: (input_size, batch_size) @ (batch_size, output_size)
        self.grad_W = self.input.T @ grad_output
        # Gradient w.r.t. biases: sum over batch, 1D array
        self.grad_b = np.sum(grad_output, axis=0)
        # Gradient for previous layer
        grad_input = grad_output @ self.W.T
        return grad_input
