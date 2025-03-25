# 模型部分应允许自定义隐藏层大小、激活函数类型，支持通过反向传播计算给定损失的梯度，不能使用pytorch或tensorflow等深度学习框架

import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.activation = self.get_activation(activation)
        self.weights = self.initialize_weights()

    def get_activation(self, activation_type):
        if activation_type == 'relu':
            return self.relu
        elif activation_type == 'sigmoid':
            return self.sigmoid
        else:
            raise ValueError("Unsupported activation function")

    def initialize_weights(self):
        weights = []
        layers = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(len(layers) - 1):
            fan_in = layers[i]
            fan_out = layers[i + 1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            weight = np.random.normal(loc=0.0, scale=std, size=(fan_in, fan_out))
            weights.append(weight)
        return weights


    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        activations = [x]
        for weight in self.weights[:-1]:
            x = np.dot(x, weight)
            x = self.activation(x)
            activations.append(x)
        x = np.dot(x, self.weights[-1])
        activations.append(x)
        return activations

    def backward(self, activations, loss_gradient):
        gradients = []
        dActivations = activations[:]
        dActivations[-1] = loss_gradient
        for i in range(len(self.weights) - 1, -1, -1):
            gradients.append(np.dot(activations[i].T, dActivations[i + 1]))
            if i != 0:
                dActivations[i] = np.multiply(np.dot(dActivations[i + 1], self.weights[i].T), self.activation_derivative(activations[i]))

        gradients.reverse()
        return gradients

    def activation_derivative(self, x):
        if self.activation == self.relu:
            return np.where(x > 0, 1, 0)
        elif self.activation == self.sigmoid:
            return self.sigmoid(x) * (1 - self.sigmoid(x))

    def update_weights(self, gradients, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i]

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            activations = self.forward(x)
            loss = self.compute_loss(y, activations[-1])
            loss_gradient = 2 * (activations[-1] - y) / len(y)
            gradients = self.backward(activations, loss_gradient)
            self.update_weights(gradients, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x):
        activations = self.forward(x)
        return activations[-1]

# 示例使用
if __name__ == "__main__":
    mlp = MLP(input_size=3, hidden_sizes=[4, 4], output_size=1, activation='relu')
    x_train = np.random.randn(10, 3)
    y_train = np.random.randn(10, 1)
    mlp.train(x_train, y_train, epochs=1000, learning_rate=0.01)
    predictions = mlp.predict(x_train)
    print("Predictions:", predictions)
