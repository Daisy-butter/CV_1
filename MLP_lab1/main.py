import numpy as np
import os
import pickle
from MLP_model import MLP
from train import CIFAR10Loader, Trainer
from config import Hyperparameters as hp


def load_cifar10_local(data_path):
    """Load CIFAR-10 dataset from a local path and return train, validation, and test datasets."""
    def unpickle(file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    # Load training data batches
    X_train = []
    y_train = []
    for batch in range(1, 6):  # CIFAR-10 training batches are 'data_batch_1' to 'data_batch_5'
        batch_file = os.path.join(data_path, f"data_batch_{batch}")
        batch_data = unpickle(batch_file)
        X_train.append(batch_data[b"data"])
        y_train.append(batch_data[b"labels"])
    X_train = np.vstack(X_train).astype(np.float32)
    y_train = np.hstack(y_train).astype(np.int64)

    # Load test data
    test_data = unpickle(os.path.join(data_path, "test_batch"))
    X_test = test_data[b"data"].astype(np.float32)
    y_test = np.array(test_data[b"labels"], dtype=np.int64)

    # Normalize data to [0, 1]
    X_train /= 255.0
    X_test /= 255.0

    return X_train, y_train, X_test, y_test


def main():
    # Local CIFAR-10 path
    data_path = r"C:\Users\31521\OneDrive\桌面\files\academic\FDU\25春大三下\计算机视觉\lab_data\cifar-10-batches-py"

    # Load local CIFAR-10 data
    X_train, y_train, X_test, y_test = load_cifar10_local(data_path)

    # Split training data into training and validation sets (90% train, 10% validation)
    num_val = int(0.1 * len(X_train))
    X_val, y_val = X_train[:num_val], y_train[:num_val]
    X_train, y_train = X_train[num_val:], y_train[num_val:]

    # Flatten image data for MLP input
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_val = X_val.reshape(X_val.shape[0], -1)

    # Initialize the model
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
