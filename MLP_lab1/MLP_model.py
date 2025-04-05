# 模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度

import numpy as np
from config import Hyperparameters as hp

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        valid_activations = ['relu', 'leaky_relu', 'sigmoid']
        if activation not in valid_activations:
            raise ValueError(f"Invalid activation function. Choose from {valid_activations}.")
        
        # Initialize the model with given sizes and activation function
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = activation
        
        self.layers = len(hidden_sizes) + 1
        self.weights = []
        self.biases = []
        
        sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(sizes) - 1):
            # Initialize weights with small random values. For leaky-relu, use He initialization
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * np.sqrt(2 / sizes[i]))
            # Initialize biases with zeros
            self.biases.append(np.zeros((1, sizes[i + 1])))
    
    def _activate(self, x):
        # Activation function(relu, leaky_relu, sigmoid)
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)  # Leaky ReLU with a slope of 0.01
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        else:
            raise ValueError(f'Unknown activation function: {self.activation}')

    def _activate_derivative(self, x):
        # Derivative of activation function
        if self.activation == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)  # Derivative of Leaky ReLU
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        else:
            raise ValueError(f'Unknown activation function: {self.activation}')
        
    def softmax(self, x):
        # Apply softmax function
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, x):
        self.inputs = []  # Store pre-activation (z values)
        self.activations = [x]  # Store post-activation values (start with input layer)

        current_activation = x
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(current_activation, w) + b
            self.inputs.append(z)
            current_activation = self._activate(z)  # Apply activation
            self.activations.append(current_activation)

        # Output layer (no activation, just linear transformation)
        z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.inputs.append(z)
        return z  # Return logits for classification

    def backward(self, logits, y):
        batch_size = y.shape[0]
        dz = self.softmax(logits)  # Apply softmax and compute gradient
        dz[range(batch_size), y] -= 1
        dz /= batch_size

        d_weights = []
        d_biases = []

        # Backward pass through layers
        for i in reversed(range(self.layers)):
            dw = np.dot(self.activations[i].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            d_weights.insert(0, dw)  # Insert grad weights at the beginning
            d_biases.insert(0, db)

            if i > 0:  # For non-output layers, backpropagate gradients
                dz = np.dot(dz, self.weights[i].T) * self._activate_derivative(self.inputs[i - 1])

        return d_weights, d_biases
