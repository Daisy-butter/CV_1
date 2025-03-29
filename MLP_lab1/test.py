import numpy as np
import os
import pickle
from MLP_model import MLP
from config import Hyperparameters as hp


def load_cifar10_local(data_path):
    """Load CIFAR-10 dataset from a local path and return test dataset."""
    def unpickle(file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    # Load test data
    test_data = unpickle(os.path.join(data_path, "test_batch"))
    X_test = test_data[b"data"].astype(np.float32)
    y_test = np.array(test_data[b"labels"], dtype=np.int64)

    # Normalize test data to [-1, 1]
    X_test /= 255.0
    X_test = 2 * X_test - 1

    return X_test, y_test


class Tester:
    def __init__(self, model):
        self.model = model

    def softmax(self, x):
        # Apply softmax function
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
        return exps / np.sum(exps, axis=1, keepdims=True)

    def compute_accuracy(self, logits, y):
        # Compute accuracy
        y_pred = np.argmax(self.softmax(logits), axis=1)
        return np.mean(y_pred == y)

    def test(self, X_test, y_test):
        # Forward pass on test data
        logits = self.model.forward(X_test)
        # Compute accuracy on test set
        accuracy = self.compute_accuracy(logits, y_test)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy


def main():
    # Local CIFAR-10 path
    data_path = r"C:\Users\31521\OneDrive\桌面\files\academic\FDU\25春大三下\计算机视觉\lab_data\cifar-10-batches-py"

    # Load local CIFAR-10 test data
    X_test, y_test = load_cifar10_local(data_path)

    # Flatten test data
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Define model structure (same as used during training)
    input_size = 32 * 32 * 3
    hidden_sizes = hp.HIDDEN_SIZES  # Example hidden layer sizes
    output_size = 10  # CIFAR-10 has 10 classes
    activation = 'leaky_relu'
    model = MLP(input_size, hidden_sizes, output_size, activation)  # Example activation function

    ### Load trained model weights and biases from `best_model` ###
    load_dir = "best_model"  # Directory containing the best trained model
    try:
        # Initialize empty lists for weights and biases
        weights = []
        biases = []

        # Load each layer's weights and biases
        num_layers = len(hidden_sizes) + 1  # Hidden layers + output layer
        for layer_idx in range(num_layers):
            weight_path = os.path.join(load_dir, f'weights_layer_{layer_idx}_epoch_best.npy')
            bias_path = os.path.join(load_dir, f'biases_layer_{layer_idx}_epoch_best.npy')

            weights.append(np.load(weight_path, allow_pickle=True))
            biases.append(np.load(bias_path, allow_pickle=True))

        # Assign loaded weights and biases to the model
        model.weights = weights
        model.biases = biases
        print(f"Best model loaded successfully from {load_dir}")
    except FileNotFoundError:
        print(f"Error: Best model files not found in {load_dir}. Please train the model and save it first!")
        return

    # Initialize tester
    tester = Tester(model)

    # Test the model on the test set
    tester.test(X_test, y_test)


if __name__ == "__main__":
    main()
