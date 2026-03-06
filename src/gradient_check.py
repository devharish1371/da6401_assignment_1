"""
Gradient Consistency Check — Pure NumPy, no Keras required.
Verifies analytical gradients against numerical gradients.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
np.random.seed(42)

from src.ann.neural_layer import DenseLayer
from src.ann.activations import Sigmoid, Tanh, ReLU, Softmax
from src.ann.objective_functions import CrossEntropyLoss, MSELoss


def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient using central differences."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]
        x[idx] = old_val + h
        fx_plus = f()
        x[idx] = old_val - h
        fx_minus = f()
        grad[idx] = (fx_plus - fx_minus) / (2 * h)
        x[idx] = old_val
        it.iternext()
    return grad


def check_gradients(activation_name, loss_name):
    """Check gradients for a 2-layer network with given activation and loss."""
    batch_size = 3
    input_size = 5
    hidden_size = 4
    output_size = 3

    # Build small network
    layer1 = DenseLayer(input_size, hidden_size, weight_init='xavier')
    if activation_name == 'sigmoid':
        act = Sigmoid()
    elif activation_name == 'tanh':
        act = Tanh()
    else:
        act = ReLU()

    layer2 = DenseLayer(hidden_size, output_size, weight_init='xavier')
    softmax = Softmax()

    if loss_name == 'cross_entropy':
        loss_fn = CrossEntropyLoss()
    else:
        loss_fn = MSELoss()

    X = np.random.randn(batch_size, input_size)
    y_true = np.zeros((batch_size, output_size))
    y_true[np.arange(batch_size), np.random.randint(0, output_size, batch_size)] = 1.0

    # Forward + backward
    h1 = layer1.forward(X)
    a1 = act.forward(h1)
    h2 = layer2.forward(a1)
    y_pred = softmax.forward(h2)
    loss = loss_fn.forward(y_true, y_pred)

    grad = loss_fn.backward(y_true, y_pred)
    grad = softmax.backward(grad)
    grad = layer2.backward(grad)
    grad = act.backward(grad)
    grad = layer1.backward(grad)

    # Numerical check for each parameter
    results = []

    def compute_loss():
        h1_ = layer1.forward(X)
        a1_ = act.forward(h1_)
        h2_ = layer2.forward(a1_)
        yp_ = softmax.forward(h2_)
        return loss_fn.forward(y_true, yp_)

    for name, layer in [("Layer2", layer2), ("Layer1", layer1)]:
        num_gW = numerical_gradient(compute_loss, layer.W)
        num_gb = numerical_gradient(compute_loss, layer.b)
        diff_W = np.max(np.abs(num_gW - layer.grad_W))
        diff_b = np.max(np.abs(num_gb - layer.grad_b))
        results.append((f"{name}.grad_W", diff_W))
        results.append((f"{name}.grad_b", diff_b))

    return results


print("=" * 60)
print("  GRADIENT CONSISTENCY CHECK")
print("=" * 60)

all_pass = True
for act_name in ['sigmoid', 'tanh', 'relu']:
    for loss_name in ['cross_entropy', 'mean_squared_error']:
        print(f"\n--- {act_name} + {loss_name} ---")
        results = check_gradients(act_name, loss_name)
        for param_name, diff in results:
            passed = diff < 1e-5  # Use 1e-5 for numerical grad tolerance
            status = "PASS ✓" if passed else "FAIL ✗"
            print(f"  {param_name}: diff={diff:.2e}  {status}")
            if not passed:
                all_pass = False

print("\n" + "=" * 60)
print(f"  OVERALL: {'ALL PASSED ✓' if all_pass else 'SOME FAILED ✗'}")
print("=" * 60)
