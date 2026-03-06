"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""
import numpy as np


def load_data(dataset_name="fashion_mnist", val_split=0.1):
    """
    Load and preprocess dataset.

    Args:
        dataset_name: 'mnist' or 'fashion_mnist'
        val_split: fraction of training data to use for validation

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
        - X shapes: (N, 784) float64, values in [0, 1]
        - y shapes: (N, 10) one-hot float64
    """
    from keras.datasets import mnist, fashion_mnist

    if dataset_name == "mnist":
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnist' or 'fashion_mnist'.")

    # Flatten images: (N, 28, 28) -> (N, 784)
    X_train_full = X_train_full.reshape(X_train_full.shape[0], -1).astype(np.float64)
    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float64)

    # Normalize to [0, 1]
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0

    # One-hot encode labels
    num_classes = 10
    y_train_full_oh = np.zeros((y_train_full.shape[0], num_classes), dtype=np.float64)
    y_train_full_oh[np.arange(y_train_full.shape[0]), y_train_full] = 1.0

    y_test_oh = np.zeros((y_test.shape[0], num_classes), dtype=np.float64)
    y_test_oh[np.arange(y_test.shape[0]), y_test] = 1.0

    # Split into train and validation
    num_val = int(X_train_full.shape[0] * val_split)
    indices = np.random.permutation(X_train_full.shape[0])
    val_idx, train_idx = indices[:num_val], indices[num_val:]

    X_val = X_train_full[val_idx]
    y_val = y_train_full_oh[val_idx]
    X_train = X_train_full[train_idx]
    y_train = y_train_full_oh[train_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test_oh
