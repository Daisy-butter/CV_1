# 训练部分应实现 SGD 优化器、学习率下降、交叉熵损失和 L2 正则化，并能根据验证集指标自动保存最优的模型权重

import numpy as np
import os
import pickle
import urllib.request
import tarfile
from MLP_model import MLP
from config import Hyperparameters as hp

class CIFAR10Loader:
    def __init__(self, data_dir='cifar-10-batches-py'):
        self.data_dir = data_dir
    
    def download_and_extract(self, url):
        # Download and extract the dataset
        file_name = url.split('/')[-1]
        if not os.path.exists(file_name):
            print(f"Downloading {file_name}...")
            urllib.request.urlretrieve(url, file_name)
        if not os.path.exists(self.data_dir):
            print("Extracting dataset...")
            with tarfile.open(file_name) as tar:
                tar.extractall()
    
    def load_data(self):
        # Load CIFAR-10 dataset from disk
        def unpickle(file):
            with open(file, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')
        
        train_data = []
        train_labels = []
        for i in range(1, 6):
            batch = unpickle(os.path.join(self.data_dir, f'data_batch_{i}'))
            train_data.append(batch[b'data'])
            train_labels += batch[b'labels']
        
        test_batch = unpickle(os.path.join(self.data_dir, 'test_batch'))
        test_data = test_batch[b'data']
        test_labels = test_batch[b'labels']
        
        train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        train_labels = np.array(train_labels)
        test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_labels = np.array(test_labels)
        
        return train_data, train_labels, test_data, test_labels

    def preprocess(self, X, y):
        # Normalize the data and convert labels to one-hot encoding
        X = X.astype('float32') / 255.0  # Normalize to [0, 1]
        return X, y

class Trainer:
    def __init__(self, model, learning_rate=hp.LEARNING_RATE, lr_decay=hp.LR_DECAY, reg_strength=hp.REG_STRENGTH):
        self.model = model
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.reg_strength = reg_strength
        self.best_val_loss = float('inf')
        self.best_weights = None
        self.best_biases = None

    def cross_entropy_loss(self, logits, y):
        # Calculate cross-entropy loss
        batch_size = y.shape[0]
        probs = self.model.softmax(logits)  # Apply softmax to logits
        log_probs = np.log(probs[range(batch_size), y])
        data_loss = -np.sum(log_probs) / batch_size
        return data_loss

    def compute_loss(self, logits, y):
        # Compute loss including regularization
        data_loss = self.cross_entropy_loss(logits, y)
        reg_loss = 0
        for w in self.model.weights:
            reg_loss += self.reg_strength * np.sum(w**2)
        return data_loss + reg_loss
    
    def compute_accuracy(self, logits, y):
        probs = self.model.softmax(logits)
        y_pred = np.argmax(probs, axis=1)
        accuracy = np.mean(y_pred == y)
        return accuracy

    def update_weights(self, gradient_weights, gradient_biases):
        for j in range(len(self.model.weights)):  # 遍历每一层的权重
            self.model.weights[j] -= self.learning_rate * (gradient_weights[j] + self.reg_strength * self.model.weights[j])  # L2 正则化
            self.model.biases[j] -= self.learning_rate * gradient_biases[j]

    def train(self, X_train, y_train, X_val, y_val, epochs=hp.EPOCHS, batch_size=hp.BATCH_SIZE, print_every=hp.PRINT_EVERY):
        num_train = X_train.shape[0]
        for epoch in range(epochs):
            perm = np.random.permutation(num_train) # Shuffle the training data
            X_train = X_train[perm]
            y_train = y_train[perm]

            # Split into batches
        for i in range(0, num_train, batch_size):
            # Mini-batch数据
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
    
            # Forward pass
            logits = self.model.forward(X_batch)
    
            # Compute loss and gradients
            loss = self.compute_loss(logits, y_batch)
            gradient_weights, gradient_biases = self.model.backward(logits, y_batch)
    
            # Update weights and biases
            self.update_weights(gradient_weights, gradient_biases)  # 封装更新逻辑

        # 每个epoch后，验证集评估
        val_logits = self.model.forward(X_val)
        val_loss = self.compute_loss(val_logits, y_val)
        # val_accuracy = self.compute_accuracy(val_logits, y_val)

        # 学习率衰减
        hp.update_learning_rate()

        # 保存当前模型
        self.save_model(epoch, save_dir="saved_models")

        # 保存最佳模型
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_model(epoch="best", save_dir=hp.MODEL_SAVE_DIR, is_best=True)

        # 打印日志
        if epoch % print_every == 0:
            print(f"Epoch {epoch} - Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")

       
    def save_model(self, epoch, save_dir="saved_models", is_best=False):
        if is_best:
            save_dir = "best_model"  # 如果是最佳模型，则强制保存到 `best_model`
    
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        np.save(os.path.join(save_dir, f'weights_epoch_{epoch}.npy'), self.model.weights)
        np.save(os.path.join(save_dir, f'biases_epoch_{epoch}.npy'), self.model.biases)
        print(f"Model for epoch {epoch} saved to {save_dir}")
   
    def load_model(self, load_dir=hp.MODEL_SAVE_DIR):
        # Load model weights and biases
        self.model.weights = np.load(os.path.join(load_dir, 'weights.npy'), allow_pickle=True)
        self.model.biases = np.load(os.path.join(load_dir, 'biases.npy'), allow_pickle=True)
        print(f"Model loaded from {load_dir}")

