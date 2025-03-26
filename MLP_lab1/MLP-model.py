# 模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
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
            # Initialize weights with small random values
            self.weights.append(np.random.randn(sizes[i], sizes[i + 1]) * 0.01)
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

    def forward(self, x):
        """
        Forward pass.

        :param x: Input data, shape (batch_size, input_size).
        :return: Output predictions.
        """
        self.inputs = []  # Store intermediate inputs
        self.outputs = []  # Store intermediate outputs

        # Forward through layers
        current_activation = x
        self.inputs.append(current_activation)

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(current_activation, w) + b
            current_activation = self._activate(z)
            self.inputs.append(z)
            self.outputs.append(current_activation)

        # Final output layer (softmax)
        z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.inputs.append(z)
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        final_output = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        self.outputs.append(final_output)

        return final_output

    def backward(self, x, y, learning_rate=0.01, reg_lambda=0.01):
        """
        Backward pass (computing gradients and updating parameters).

        :param x: Input data, shape (batch_size, input_size).
        :param y: True labels, one-hot encoded, shape (batch_size, output_size).
        :param learning_rate: Learning rate for SGD.
        :param reg_lambda: L2 regularization parameter.
        """
        # Compute loss gradient (cross-entropy loss)
        batch_size = y.shape[0]
        final_output = self.outputs[-1]
        dz = final_output - y  # Loss derivative for the last layer

        # Backward through layers
        for i in reversed(range(self.layers)):
            dw = np.dot(self.inputs[i].T, dz) / batch_size + reg_lambda * self.weights[i] / batch_size
            db = np.sum(dz, axis=0, keepdims=True) / batch_size

            # Update parameters
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

            if i > 0:  # Don't compute delta for input layer
                dz = np.dot(dz, self.weights[i].T) * self._activate_derivative(self.inputs[i])

    def compute_loss(self, y_pred, y_true, reg_lambda):
        """
        Compute the loss function value (cross-entropy + L2 regularization).

        :param y_pred: Predictions from the model.
        :param y_true: Ground truth labels, one-hot encoded.
        :param reg_lambda: L2 regularization parameter.
        :return: Loss value.
        """
        batch_size = y_true.shape[0]
        cross_entropy_loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / batch_size
        l2_regularization = reg_lambda / 2 * sum(np.sum(w ** 2) for w in self.weights)
        return cross_entropy_loss + l2_regularization
