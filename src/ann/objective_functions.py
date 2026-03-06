"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""
import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy Loss.
    Forward: L = -mean(sum(y_true * log(y_pred + eps)))
    Backward: dL/dy_pred = -y_true / (y_pred + eps) / batch_size
    """

    def forward(self, y_true, y_pred):
        """
        Compute cross-entropy loss.

        Args:
            y_true: one-hot encoded labels (batch_size, num_classes)
            y_pred: output probabilities (batch_size, num_classes)

        Returns:
            Scalar loss value
        """
        eps = 1e-12
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
        return loss

    def backward(self, y_true, y_pred):
        """
        Compute gradient of loss w.r.t. y_pred (modular gradient).

        Args:
            y_true: one-hot encoded labels (batch_size, num_classes)
            y_pred: output probabilities (batch_size, num_classes)

        Returns:
            Gradient (batch_size, num_classes)
        """
        batch_size = y_true.shape[0]
        eps = 1e-12
        return -y_true / (y_pred + eps) / batch_size


class MSELoss:
    """
    Mean Squared Error Loss.
    Forward: L = (1/2) * mean(sum((y_true - y_pred)^2))
    Backward: dL/dy_pred = (y_pred - y_true) / batch_size
    """

    def forward(self, y_true, y_pred):
        """
        Compute MSE loss.

        Args:
            y_true: one-hot encoded labels (batch_size, num_classes)
            y_pred: output probabilities (batch_size, num_classes)

        Returns:
            Scalar loss value
        """
        loss = 0.5 * np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
        return loss

    def backward(self, y_true, y_pred):
        """
        Compute gradient of MSE loss w.r.t. y_pred.

        Args:
            y_true: one-hot encoded labels (batch_size, num_classes)
            y_pred: output probabilities (batch_size, num_classes)

        Returns:
            Gradient dL/dy_pred (batch_size, num_classes)
        """
        batch_size = y_true.shape[0]
        return (y_pred - y_true) / batch_size


def get_loss(name):
    """
    Factory function to get loss by name.

    Args:
        name: 'cross_entropy' or 'mean_squared_error'

    Returns:
        Loss class instance
    """
    losses = {
        "cross_entropy": CrossEntropyLoss,
        "mean_squared_error": MSELoss,
    }
    name = name.lower()
    if name not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
    return losses[name]()
