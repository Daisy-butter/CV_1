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
        self.inputs = []  # Store intermediate inputs
        self.outputs = []  # Store intermediate outputs
    # Forward through layers
        current_activation = x
        self.inputs.append(current_activation)

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(current_activation, w) + b
            current_activation = self._activate(z)
            self.inputs.append(z)  # Store intermediate results
            self.outputs.append(current_activation)
    # Final output layer (logits, no softmax here, will be applied with cross entropy loss in train.py)
        z = np.dot(current_activation, self.weights[-1]) + self.biases[-1]
        self.inputs.append(z)  # Store logits
        self.outputs.append(z)  # Append logits
    # Return logits instead of applying softmax here
        return z
    
    def backward(self, x, y):
        batch_size = y.shape[0]  # batch_size是y的行数，实现SGD时可能需要修改
        final_output = self.outputs[-1]  # 获取最后一层的输出
        dz = final_output - y  # Loss derivative for the last layer

        d_weights = []
        d_biases = []

    # Backward through layers to calculate gradients
        for i in reversed(range(self.layers)):
            dw = np.dot(self.inputs[i].T, dz) / batch_size  # Gradient for weights
            db = np.sum(dz, axis=0, keepdims=True) / batch_size  # Gradient for biases

            d_weights.insert(0, dw)
            d_biases.insert(0, db)

            if i > 0:  # Compute dz for the previous layer
                dz = np.dot(dz, self.weights[i].T) * self._activate_derivative(self.inputs[i])

        return d_weights, d_biases
