"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import os
import json
import numpy as np

from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')

    parser.add_argument('--model_path', type=str, default='models/best_model.npy',
                        help='Path to saved model weights')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to model config JSON (auto-detected if not given)')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to evaluate on (default: fashion_mnist)')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('-nhl', '--num_layers', type=int, default=None,
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=None,
                        help='Neurons per hidden layer (list)')
    parser.add_argument('-a', '--activation', type=str, default=None,
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='Activation function')
    parser.add_argument('-wi', '--weight_init', type=str, default=None,
                        choices=['random', 'xavier'],
                        help='Weight initialization method')
    parser.add_argument('-l', '--loss', type=str, default=None,
                        choices=['cross_entropy', 'mean_squared_error'],
                        help='Loss function')

    return parser.parse_args()


def load_model(model_path, config):
    """
    Load trained model from disk.

    Args:
        model_path: path to .npy weights file
        config: namespace or dict with model architecture info

    Returns:
        NeuralNetwork instance with loaded weights
    """
    model = NeuralNetwork(config)
    model.load(model_path)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.

    Returns:
        Dictionary with: logits, loss, accuracy, f1, precision, recall
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    # Forward pass
    logits = model.predict(X_test)
    loss = model.loss_fn.forward(y_test, logits)

    y_pred_labels = np.argmax(logits, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_labels == y_true_labels)
    f1 = f1_score(y_true_labels, y_pred_labels, average='macro')
    precision = precision_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)
    recall = recall_score(y_true_labels, y_pred_labels, average='macro', zero_division=0)

    return {
        "logits": logits,
        "loss": float(loss),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


def main():
    """
    Main inference function.

    Returns:
        Dictionary with: logits, loss, accuracy, f1, precision, recall
    """
    args = parse_arguments()

    # Try to load config from JSON
    config_path = args.config_path
    if config_path is None:
        # Auto-detect: look for best_config.json in same directory as model
        model_dir = os.path.dirname(args.model_path) or '.'
        config_path = os.path.join(model_dir, 'best_config.json')
        # Fallback to old naming
        if not os.path.exists(config_path):
            config_path = args.model_path.replace('.npy', '_config.json')

    config_dict = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        print(f"Loaded config from {config_path}")

    # Build a namespace for NeuralNetwork constructor, CLI args override config
    class Config:
        pass

    config = Config()
    config.dataset = args.dataset or config_dict.get('dataset', 'fashion_mnist')
    config.num_layers = args.num_layers or config_dict.get('num_layers', 3)
    config.hidden_size = args.hidden_size or config_dict.get('hidden_size', 128)
    config.activation = args.activation or config_dict.get('activation', 'relu')
    config.weight_init = args.weight_init or config_dict.get('weight_init', 'xavier')
    config.loss = args.loss or config_dict.get('loss', 'cross_entropy')
    config.optimizer = config_dict.get('optimizer', 'adam')
    config.learning_rate = config_dict.get('learning_rate', 0.001)
    config.weight_decay = config_dict.get('weight_decay', 0.0)
    config.batch_size = args.batch_size
    config.epochs = config_dict.get('epochs', 10)

    # Load data
    print(f"Loading {config.dataset} test data...")
    _, _, _, _, X_test, y_test = load_data(config.dataset)
    print(f"  Test set: {X_test.shape}")

    # Build and load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, config)

    # Evaluate
    print("Evaluating...")
    results = evaluate_model(model, X_test, y_test)

    print(f"\n{'='*40}")
    print(f"  Loss:      {results['loss']:.4f}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  F1-Score:  {results['f1']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"{'='*40}")

    print("\nEvaluation complete!")
    return results


if __name__ == '__main__':
    main()
