"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
import os

from src.ann.neural_layer import DenseLayer
from src.ann.activations import get_activation
from src.ann.objective_functions import get_loss
from src.ann.optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.

    Architecture: Input → [Dense → Activation] × num_layers → Dense → Softmax → Output

    Gradient storage convention (for autograder):
        self.grad_W and self.grad_b are NumPy object arrays where
        index 0 corresponds to the LAST (output) layer.
    """

    def __init__(self, cli_args):
        """
        Build the network from CLI arguments (or equivalent dict/namespace).

        Expected attributes on cli_args:
            - dataset: str
            - num_layers: int (number of hidden layers)
            - hidden_size: int (neurons per hidden layer)
            - activation: str ('sigmoid', 'tanh', 'relu')
            - weight_init: str ('random', 'xavier')
            - optimizer: str
            - learning_rate: float
            - weight_decay: float
            - loss: str ('cross_entropy', 'mean_squared_error')
            - batch_size: int
            - epochs: int
        """
        # Parse config
        num_layers = getattr(cli_args, 'num_layers', 3)
        hidden_size = getattr(cli_args, 'hidden_size', [128])
        activation_name = getattr(cli_args, 'activation', 'relu')
        weight_init = getattr(cli_args, 'weight_init', 'xavier')
        optimizer_name = getattr(cli_args, 'optimizer', 'adam')
        lr = getattr(cli_args, 'learning_rate', 0.001)
        weight_decay = getattr(cli_args, 'weight_decay', 0.0)
        loss_name = getattr(cli_args, 'loss', 'cross_entropy')

        input_size = getattr(cli_args, 'input_size', 784)
        output_size = getattr(cli_args, 'output_size', 10)

        # Normalize hidden_size to a list of per-layer sizes
        if isinstance(hidden_size, int):
            layer_sizes = [hidden_size] * num_layers
        elif isinstance(hidden_size, (list, tuple)):
            if len(hidden_size) >= num_layers:
                layer_sizes = list(hidden_size[:num_layers])
            else:
                # Pad with last value if list is shorter than num_layers
                layer_sizes = list(hidden_size) + [hidden_size[-1]] * (num_layers - len(hidden_size))
        else:
            layer_sizes = [128] * num_layers

        # Build layers and activations
        self.layers = []        # DenseLayer instances
        self.activations = []   # Activation instances (one per hidden layer + output softmax)

        # Hidden layers
        prev_size = input_size
        for i in range(num_layers):
            sz = layer_sizes[i]
            layer = DenseLayer(prev_size, sz, weight_init=weight_init)
            self.layers.append(layer)
            self.activations.append(get_activation(activation_name))
            prev_size = sz

        # Output layer
        output_layer = DenseLayer(prev_size, output_size, weight_init=weight_init)
        self.layers.append(output_layer)
        self.activations.append(get_activation('softmax'))

        # Loss and optimizer
        self.loss_fn = get_loss(loss_name)
        self.optimizer = get_optimizer(optimizer_name, self.layers,
                                       lr=lr, weight_decay=weight_decay)

        # Gradient storage (set during backward)
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits (no softmax applied)
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        out = X
        # Forward through all but the last activation (softmax)
        for i in range(len(self.layers)):
            out = self.layers[i].forward(out)
            if i < len(self.layers) - 1:
                out = self.activations[i].forward(out)
        return out

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        # Compute loss gradient (dL/dy_pred)
        grad = self.loss_fn.backward(y_true, y_pred)

        # Populate state for the last activation (usually Softmax) because
        # forward() returns logits and skips the last activation's forward call.
        self.activations[-1].output = y_pred

        grad_W_list = []
        grad_b_list = []

        # Backprop through layers in reverse; collect grads so that index 0 = last layer
        for i in reversed(range(len(self.layers))):
            # Backprop through activation (softmax for last layer, activation for others)
            grad = self.activations[i].backward(grad)
            # Backprop through dense layer (sets layer.grad_W, layer.grad_b)
            grad = self.layers[i].backward(grad)
            
            # Incorporate weight decay into analytical gradient if present
            # Gradient of L2 term 0.5 * lambda * W^2 is lambda * W
            wd_term = 0.0
            if hasattr(self, 'optimizer') and self.optimizer.weight_decay > 0:
                wd_term = self.optimizer.weight_decay * self.layers[i].W
            
            grad_W_list.append(self.layers[i].grad_W + wd_term)
            grad_b_list.append(self.layers[i].grad_b)

        # create explicit object arrays to avoid numpy trying to broadcast shapes
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        print("Shape of grad_Ws:", self.grad_W.shape, self.grad_W[1].shape)
        print("Shape of grad_bs:", self.grad_b.shape, self.grad_b[1].shape)
        return self.grad_W, self.grad_b

    def update_weights(self):
        """Apply optimizer step to update all layer weights."""
        self.optimizer.step()

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=10, batch_size=64, wandb_log=False):
        """
        Mini-batch training loop.

        Args:
            X_train: training data (N, 784)
            y_train: training labels one-hot (N, 10)
            X_val: validation data (optional)
            y_val: validation labels (optional)
            epochs: number of training epochs
            batch_size: mini-batch size
            wandb_log: whether to log to W&B

        Returns:
            dict with training history
        """
        history = {"train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": []}
        n = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            # Shuffle training data
            perm = np.random.permutation(n)
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]

            epoch_loss = 0.0
            epoch_correct = 0

            num_batches = int(np.ceil(n / batch_size))
            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                logits = self.forward(X_batch)
                y_pred = self.activations[-1].forward(logits)

                # Compute loss
                loss = self.loss_fn.forward(y_batch, y_pred)

                # Add L2 regularization to the loss for logging
                if self.optimizer.weight_decay > 0:
                    l2_reg = sum(np.sum(layer.W ** 2) for layer in self.layers)
                    loss += 0.5 * self.optimizer.weight_decay * l2_reg

                epoch_loss += loss * (end - start)
                epoch_correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))

                # Backward pass
                self.backward(y_batch, y_pred)

                # Update weights
                self.update_weights()

            # Epoch metrics
            train_loss = epoch_loss / n
            train_acc = epoch_correct / n
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            val_loss, val_acc = 0.0, 0.0
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

            # Logging
            log_msg = (f"Epoch {epoch}/{epochs} - "
                       f"train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}")
            if X_val is not None:
                log_msg += f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}"
            print(log_msg)

            # W&B logging
            if wandb_log:
                try:
                    import wandb
                    log_dict = {"epoch": epoch,
                                "train_loss": train_loss,
                                "train_acc": train_acc}
                    if X_val is not None:
                        log_dict["val_loss"] = val_loss
                        log_dict["val_acc"] = val_acc
                    wandb.log(log_dict)
                except ImportError:
                    pass

        return history

    def evaluate(self, X, y):
        """
        Evaluate model on data.

        Args:
            X: data (N, 784)
            y: labels one-hot (N, 10)

        Returns:
            (loss, accuracy) tuple
        """
        logits = self.forward(X)
        y_pred = self.activations[-1].forward(logits)
        loss = self.loss_fn.forward(y, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
        return loss, accuracy

    def predict(self, X):
        """Return softmax output probabilities."""
        logits = self.forward(X)
        return self.activations[-1].forward(logits)

    def get_weights(self):
        """Get all weights as a dictionary."""
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        """Set weights from a dictionary."""
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()

    def save(self, path):
        """Save model weights to .npy file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        np.save(path, self.get_weights(), allow_pickle=True)

    def load(self, path):
        """Load model weights from .npy file."""
        weight_dict = np.load(path, allow_pickle=True).item()
        self.set_weights(weight_dict)
