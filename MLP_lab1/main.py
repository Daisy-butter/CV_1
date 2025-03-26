import numpy as np
import os
import pickle
import urllib.request
import tarfile
from MLP_model import MLP
from train import CIFAR10Loader, Trainer
from config import Hyperparameters as hp

def main():
    # Load CIFAR-10
    data_loader = CIFAR10Loader()
    cifar_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    data_loader.download_and_extract(cifar_url)

    # Load and preprocess data
    X_train, y_train, X_test, y_test = data_loader.load_data()
    X_train, y_train = data_loader.preprocess(X_train, y_train)
    X_test, y_test = data_loader.preprocess(X_test, y_test)

    # Split training data into training and validation sets
    num_val = int(0.1 * len(X_train))
    X_val, y_val = X_train[:num_val], y_train[:num_val]
    X_train, y_train = X_train[num_val:], y_train[num_val:]

    # Initialize model
    input_size = 32 * 32 * 3
    hidden_sizes = hp.HIDDEN_SIZES  # Example hidden layer sizes
    output_size = 10  # CIFAR-10 has 10 classes
    activation = 'leaky_relu'
    model = MLP(input_size, hidden_sizes, output_size, activation)

    # Initialize trainer
    trainer = Trainer(model, learning_rate=hp.LEARNING_RATE, lr_decay=hp.LR_DECAY, reg_strength=hp.REG_STRENGTH)

    # Prepare data by flattening image inputs
    X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten image data for MLP input
    X_val = X_val.reshape(X_val.shape[0], -1)

    # Train the model, mini-batch SGD
    trainer.train(X_train, y_train, X_val, y_val, epochs=hp.EPOCHS, batch_size=hp.BATCH_SIZE, print_every=hp.PRINT_EVERY)

    print("Training completed. Model weights and biases have been saved.")

if __name__ == "__main__":
    main()
