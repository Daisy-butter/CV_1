# 训练部分应实现 SGD 优化器、学习率下降、交叉熵损失和 L2 正则化，并能根据验证集指标自动保存最优的模型权重

import numpy as np
import os
from MLP_model import MLP

class Trainer:
    def __init__(self, model, learning_rate=0.01, lr_decay=0.95, reg_strength=1e-4):
        self.model = model
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.reg_strength = reg_strength
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.best_biases = None

    def softmax(self, x):
        # Apply softmax function
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement
        return exps / np.sum(exps, axis=1, keepdims=True)

    def cross_entropy_loss(self, logits, y):
        # Calculate cross-entropy loss
        batch_size = y.shape[0]
        probs = self.softmax(logits)  # Apply softmax to logits
        loss = -np.sum(np.log(probs[range(batch_size), y])) / batch_size
        return loss

    def compute_loss(self, logits, y):
        # Compute loss including regularization
        data_loss = self.cross_entropy_loss(logits, y)
        reg_loss = 0
        for w in self.model.weights:
            reg_loss += self.reg_strength * np.sum(w**2)
        return data_loss + reg_loss

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=128, print_every=10):
        num_train = X_train.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(num_train)
            X_train = X_train[perm]
            y_train = y_train[perm]
            # Split into batches
            for i in range(0, num_train, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                logits = self.model.forward(X_batch)

                # Compute loss
                loss = self.compute_loss(logits, y_batch)

                # Backward pass
                gradient_weights, gradient_biases = self.model.backward(X_batch, y_batch)

                # Update weights and biases using SGD
                for j in range(self.model.layers):
                    self.model.weights[j] -= self.learning_rate * (gradient_weights[j] + self.reg_strength * self.model.weights[j])
                    self.model.biases[j] -= self.learning_rate * gradient_biases[j]

            # Evaluate on validation set
            val_logits = self.model.forward(X_val)
            val_loss = self.compute_loss(val_logits, y_val)
            val_accuracy = self.compute_accuracy(val_logits, y_val)

            # Learning rate decay
            self.learning_rate *= self.lr_decay

            # Save the best weights based on validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_weights = [w.copy() for w in self.model.weights]
                self.best_biases = [b.copy() for b in self.model.biases]

            if epoch % print_every == 0:
                print(f"Epoch {epoch} - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Restore best weights
        self.model.weights = self.best_weights
        self.model.biases = self.best_biases
        self.save_model()

    def compute_accuracy(self, logits, y):
        # Compute accuracy
        y_pred = np.argmax(self.softmax(logits), axis=1)
        return np.mean(y_pred == y)

    def save_model(self, save_dir='best_model'):
        # Save model weights and biases
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, 'weights.npy'), self.best_weights)
        np.save(os.path.join(save_dir, 'biases.npy'), self.best_biases)
        print(f"Best model saved to {save_dir}")

    def load_model(self, load_dir='best_model'):
        # Load model weights and biases
        self.model.weights = np.load(os.path.join(load_dir, 'weights.npy'), allow_pickle=True)
        self.model.biases = np.load(os.path.join(load_dir, 'biases.npy'), allow_pickle=True)
        print(f"Model loaded from {load_dir}")
