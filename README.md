# DA6401 Assignment 1: Multi-Layer Perceptron for Image Classification

A **configurable, modular MLP** built from scratch using **only NumPy** for MNIST and Fashion-MNIST classification.

## Project Structure

```
├── src/
│   ├── ann/
│   │   ├── activations.py          # Sigmoid, Tanh, ReLU, Softmax
│   │   ├── objective_functions.py  # Cross-Entropy, MSE
│   │   ├── neural_layer.py         # DenseLayer (forward/backward, grad_W/grad_b)
│   │   ├── optimizers.py           # SGD, Momentum, NAG, RMSprop, Adam, Nadam
│   │   └── neural_network.py       # NeuralNetwork (train, evaluate, save/load)
│   ├── utils/
│   │   └── data_loader.py          # Load MNIST/Fashion-MNIST via keras.datasets
│   ├── train.py                    # CLI training script
│   ├── inference.py                # CLI evaluation script
│   ├── gradient_check.py           # Numerical gradient verification
│   └── wandb_experiments.py        # All 10 W&B experiments
├── models/
│   ├── best_model.npy              # Best model weights
│   └── best_model_config.json      # Best model hyperparameters
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
python src/train.py \
  -d fashion_mnist \
  -e 10 \
  -b 64 \
  -l cross_entropy \
  -o adam \
  -lr 0.001 \
  -wd 0 \
  -nhl 3 \
  -sz 128 \
  -a relu \
  -wi xavier \
  -wp da6401-a1
```

### CLI Arguments

| Flag | Long | Description |
|------|------|-------------|
| `-d` | `--dataset` | `mnist` or `fashion_mnist` |
| `-e` | `--epochs` | Number of training epochs |
| `-b` | `--batch_size` | Mini-batch size |
| `-l` | `--loss` | `cross_entropy` or `mean_squared_error` |
| `-o` | `--optimizer` | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` |
| `-lr` | `--learning_rate` | Learning rate |
| `-wd` | `--weight_decay` | L2 regularization |
| `-nhl` | `--num_layers` | Number of hidden layers |
| `-sz` | `--hidden_size` | Neurons per layer (list, e.g., `128 64 32`) |
| `-a` | `--activation` | `sigmoid`, `tanh`, `relu` |
| `-wi` | `--weight_init` | `random` or `xavier` |

## Inference

```bash
python src/inference.py \
  --model_path models/best_model.npy \
  -d fashion_mnist
```

Returns: accuracy, precision, recall, F1-score, loss, logits.

## W&B Experiments

Run individual experiments or all at once:

```bash
# Run all experiments
python src/wandb_experiments.py --experiment all --wandb_project da6401-a1

# Run specific experiment
python src/wandb_experiments.py --experiment 2.1
python src/wandb_experiments.py --experiment 2.3
```

### Experiments List

| Section | Title | Marks |
|---------|-------|-------|
| 2.1 | Data Exploration & Class Distribution | 3 |
| 2.2 | Hyperparameter Sweep (≥100 runs) | 6 |
| 2.3 | Optimizer Showdown | 5 |
| 2.4 | Vanishing Gradient Analysis | 5 |
| 2.5 | Dead Neuron Investigation | 6 |
| 2.6 | Loss Function Comparison | 4 |
| 2.7 | Global Performance Analysis | 4 |
| 2.8 | Error Analysis (Confusion Matrix) | 5 |
| 2.9 | Weight Init & Symmetry Breaking | 7 |
| 2.10 | Fashion-MNIST Transfer Challenge | 5 |

## Gradient Verification

```bash
python src/gradient_check.py
```

Verifies analytical gradients vs numerical gradients (tolerance 10⁻⁷) for all activation × loss combinations.

## Libraries Used

- **NumPy** — all math/neural network operations
- **keras.datasets** — data loading only
- **scikit-learn** — confusion matrix, F1/precision/recall
- **matplotlib** — visualizations
- **wandb** — experiment tracking
