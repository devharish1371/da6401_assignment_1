"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import os
import json
import numpy as np

from src.utils.data_loader import load_data
from src.ann.neural_network import NeuralNetwork


def parse_arguments():
    """
    Parse command-line arguments.

    Implements all required CLI arguments per the assignment specification.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    parser.add_argument('-wp', '--wandb_project', type=str, default='da6401-a1',
                        help='W&B project name (default: da6401-a1)')
    parser.add_argument('-we', '--wandb_entity', type=str, default=None,
                        help='W&B entity name')
    parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use (default: fashion_mnist)')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Mini-batch size (default: 64)')
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mean_squared_error'],
                        help='Loss function (default: cross_entropy)')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'],
                        help='Optimizer (default: adam)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='L2 weight decay (default: 0.0)')
    parser.add_argument('-nhl', '--num_layers', type=int, default=3,
                        help='Number of hidden layers (default: 3)')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128],
                        help='Neurons per hidden layer as a list (default: [128])')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['sigmoid', 'tanh', 'relu'],
                        help='Activation function (default: relu)')
    parser.add_argument('-wi', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier'],
                        help='Weight initialization method (default: xavier)')
    parser.add_argument('--model_save_path', type=str, default='models/best_model.npy',
                        help='Path to save the best model (default: models/best_model.npy)')

    return parser.parse_args()


def main():
    """
    Main training function.

    1. Parse CLI arguments
    2. Initialize W&B
    3. Load and preprocess data
    4. Build the neural network
    5. Train with mini-batch gradient descent
    6. Save the best model and config
    """
    args = parse_arguments()

    # Initialize W&B
    wandb_log = False
    try:
        import wandb

        # Build a descriptive run name
        run_name = (f"{args.optimizer}_lr{args.learning_rate}_"
                    f"hl{args.num_layers}_sz{args.hidden_size}_"
                    f"{args.activation}_{args.weight_init}")

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "loss": args.loss,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "activation": args.activation,
                "weight_init": args.weight_init,
            },
            name=run_name,
        )
        wandb_log = True
        print(f"W&B initialized: project={args.wandb_project}, run={run_name}")
    except Exception as e:
        print(f"W&B init failed ({e}), continuing without logging.")

    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(args.dataset)
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Build model
    print(f"\nBuilding model: {args.num_layers} hidden layers × {args.hidden_size} neurons, "
          f"activation={args.activation}, weight_init={args.weight_init}")
    model = NeuralNetwork(args)

    # Train
    print(f"\nTraining for {args.epochs} epochs (batch_size={args.batch_size}, "
          f"optimizer={args.optimizer}, lr={args.learning_rate}, "
          f"weight_decay={args.weight_decay}, loss={args.loss})...\n")

    history = model.train(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        wandb_log=wandb_log,
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    if wandb_log:
        try:
            import wandb
            wandb.log({"test_loss": test_loss, "test_acc": test_acc})
        except Exception:
            pass

    # Save best model
    model.save(args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

    # Save config alongside model (PDF requires best_config.json)
    model_dir = os.path.dirname(args.model_save_path) or '.'
    config_path = os.path.join(model_dir, 'best_config.json')
    config = {
        "dataset": args.dataset,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "activation": args.activation,
        "weight_init": args.weight_init,
        "loss": args.loss,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }
    os.makedirs(model_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")

    # Finish W&B run
    if wandb_log:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
