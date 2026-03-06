"""
Numerical Gradient Consistency Check for Neural Network
Verifies that analytical gradients match finite-difference numerical gradients.
"""
import numpy as np
from src.ann.neural_network import NeuralNetwork
from src.ann.objective_functions import get_loss
from src.ann.activations import get_activation

def check_gradients():
    """
    Perform gradient check on a toy network.
    """
    print("=" * 60)
    print("  GRADIENT CONSISTENCY CHECK (Robust Modular Version)")
    print("=" * 60)

    # Toy problem setup
    input_size = 5
    hidden_size = 4
    output_size = 3
    batch_size = 2
    
    # Create a mock CLI args object
    class Args:
        pass
    args = Args()
    args.dataset = 'mnist'
    args.num_layers = 1
    args.hidden_size = [hidden_size]
    args.activation = 'relu'
    args.weight_init = 'xavier'
    args.optimizer = 'sgd'
    args.learning_rate = 0.01
    args.weight_decay = 0.1  # Set non-zero to test L2 inclusion
    args.input_size = input_size
    args.output_size = output_size
    args.wandb_project = None

    activations = ['sigmoid', 'tanh', 'relu']
    losses = ['cross_entropy', 'mean_squared_error']

    X = np.random.randn(batch_size, input_size)
    y_true = np.zeros((batch_size, output_size))
    y_true[np.arange(batch_size), np.random.randint(0, output_size, batch_size)] = 1

    epsilon = 1e-6
    tolerance = 1e-7

    for act_name in activations:
        for loss_name in losses:
            print(f"\n--- {act_name} + {loss_name} (WD=0.1) ---")
            
            args.activation = act_name
            model = NeuralNetwork(args)
            model.loss_fn = get_loss(loss_name)
            
            # 1. Analytical backward pass
            y_pred = model.activations[-1].forward(model.forward(X))
            ana_grad_W, ana_grad_b = model.backward(y_true, y_pred)
            
            # 2. Numerical check for each layer
            passed = True
            for i, layer in enumerate(model.layers):
                # Reverse index for ana_grad (since 0 is last layer in ana_grad)
                ana_idx = len(model.layers) - 1 - i
                
                # Check W
                num_grad_W = np.zeros_like(layer.W)
                it = np.nditer(layer.W, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    orig_val = layer.W[idx]
                    
                    # f(x + eps)
                    layer.W[idx] = orig_val + epsilon
                    y_p_plus = model.activations[-1].forward(model.forward(X))
                    L_plus = model.loss_fn.forward(y_true, y_p_plus)
                    # Add L2 penalty manually for numerical check
                    L_plus += 0.5 * model.optimizer.weight_decay * sum(np.sum(l.W**2) for l in model.layers)
                    
                    # f(x - eps)
                    layer.W[idx] = orig_val - epsilon
                    y_p_minus = model.activations[-1].forward(model.forward(X))
                    L_minus = model.loss_fn.forward(y_true, y_p_minus)
                    L_minus += 0.5 * model.optimizer.weight_decay * sum(np.sum(l.W**2) for l in model.layers)
                    
                    layer.W[idx] = orig_val
                    num_grad_W[idx] = (L_plus - L_minus) / (2 * epsilon)
                    it.iternext()
                
                diff_W = np.linalg.norm(ana_grad_W[ana_idx] - num_grad_W) / (np.linalg.norm(ana_grad_W[ana_idx] + num_grad_W) + 1e-12)
                
                # Check b
                num_grad_b = np.zeros_like(layer.b)
                it = np.nditer(layer.b, flags=['multi_index'], op_flags=['readwrite'])
                while not it.finished:
                    idx = it.multi_index
                    orig_val = layer.b[idx]
                    
                    layer.b[idx] = orig_val + epsilon
                    y_p_plus = model.activations[-1].forward(model.forward(X))
                    L_plus = model.loss_fn.forward(y_true, y_p_plus)
                    L_plus += 0.5 * model.optimizer.weight_decay * sum(np.sum(l.W**2) for l in model.layers)

                    layer.b[idx] = orig_val - epsilon
                    y_p_minus = model.activations[-1].forward(model.forward(X))
                    L_minus = model.loss_fn.forward(y_true, y_p_minus)
                    L_minus += 0.5 * model.optimizer.weight_decay * sum(np.sum(l.W**2) for l in model.layers)
                    
                    layer.b[idx] = orig_val
                    num_grad_b[idx] = (L_plus - L_minus) / (2 * epsilon)
                    it.iternext()
                
                # Reshape ana_grad_b if it's 1D to match num_grad_b
                ana_b = ana_grad_b[ana_idx].reshape(num_grad_b.shape)
                diff_b = np.linalg.norm(ana_b - num_grad_b) / (np.linalg.norm(ana_b + num_grad_b) + 1e-12)

                print(f"  Layer{i+1}.grad_W: diff={diff_W:.2e}  {'PASS ✓' if diff_W < tolerance else 'FAIL ✗'}")
                print(f"  Layer{i+1}.grad_b: diff={diff_b:.2e}  {'PASS ✓' if diff_b < tolerance else 'FAIL ✗'}")
                if diff_W >= tolerance or diff_b >= tolerance:
                    passed = False

    print("\n" + "=" * 60)
    print("  OVERALL: ALL PASSED ✓" if passed else "  OVERALL: SOME FAILED ✗")
    print("=" * 60)

if __name__ == "__main__":
    check_gradients()
