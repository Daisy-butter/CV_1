import numpy as np
import os
import pickle
import urllib.request
import tarfile
from MLP_model import MLP
from train import CIFAR10Loader, Trainer
from config import Hyperparameters as hp
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split


def main():
    # Load CIFAR-10 data using PyTorch's torchvision
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
    ])

    # Download and load training data
    cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    # Split data into training and validation sets (90% train, 10% validation)
    train_size = int(0.9 * len(cifar10_train))
    val_size = len(cifar10_train) - train_size
    train_data, val_data = random_split(cifar10_train, [train_size, val_size])

    # Load test data
    cifar10_test = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # Convert training, validation, and test data to DataLoader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=len(val_data), shuffle=False)
    test_loader = torch.utils.data.DataLoader(cifar10_test, batch_size=len(cifar10_test), shuffle=False)

    # Get all data and labels as NumPy arrays
    X_train, y_train = next(iter(train_loader))
    X_val, y_val = next(iter(val_loader))
    X_test, y_test = next(iter(test_loader))

    # Convert data from PyTorch tensors to NumPy arrays
    X_train = X_train.numpy()
    y_train = y_train.numpy()
    X_val = X_val.numpy()
    y_val = y_val.numpy()
    X_test = X_test.numpy()
    y_test = y_test.numpy()

    # Flatten image data for MLP input
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    # Initialize model
    input_size = 32 * 32 * 3
    hidden_sizes = hp.HIDDEN_SIZES
    output_size = 10  # CIFAR-10 has 10 classes
    activation = 'leaky_relu'
    model = MLP(input_size, hidden_sizes, output_size, activation)

    # Initialize trainer
    trainer = Trainer(model, learning_rate=hp.LEARNING_RATE, lr_decay=hp.LR_DECAY, reg_strength=hp.REG_STRENGTH)

    # Train the model using mini-batch SGD
    trainer.train(X_train, y_train, X_val, y_val, epochs=hp.EPOCHS, batch_size=hp.BATCH_SIZE, print_every=hp.PRINT_EVERY)

    print("Training completed. Model weights and biases have been saved.")

if __name__ == "__main__":
    main()
