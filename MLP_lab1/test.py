# 测试部分需支持导入训练好的模型， 输出在测试集上的分类准确率(Accuracy)

import numpy as np
import os
from MLP_model import MLP
from train import CIFAR10Loader

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
    # Load CIFAR-10
    data_loader = CIFAR10Loader()
    cifar_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    data_loader.download_and_extract(cifar_url)

    # Load and preprocess test data
    _, _, X_test, y_test = data_loader.load_data()
    X_test, y_test = data_loader.preprocess(X_test, y_test)

    # Flatten test data
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Define model structure (same as used during training)
    input_size = 32 * 32 * 3
    hidden_sizes = [128, 64]  # Example hidden layer sizes
    output_size = 10  # CIFAR-10 has 10 classes
    model = MLP(input_size, hidden_sizes, output_size, activation='relu')  # Example activation function

    # Load trained model weights and biases
    load_dir = 'best_model'  # Directory containing the trained model
    try:
        model.weights = np.load(os.path.join(load_dir, 'weights.npy'), allow_pickle=True)
        model.biases = np.load(os.path.join(load_dir, 'biases.npy'), allow_pickle=True)
        print(f"Model loaded successfully from {load_dir}")
    except FileNotFoundError:
        print(f"Error: Model files not found in {load_dir}. Please train the model first!")
        return

    # Initialize tester
    tester = Tester(model)

    # Test the model on the test set
    tester.test(X_test, y_test)


if __name__ == "__main__":
    main()
