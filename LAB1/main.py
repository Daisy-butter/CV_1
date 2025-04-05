from MLP_model import *
from config import config
from train import optimize_model_parameters
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import pickle
import numpy as np

data_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data"

X_train = np.load(f"{data_dir}/X_train.npy")
y_train = np.load(f"{data_dir}/y_train.npy")
X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

INPUT_DIM = config.INPUT_DIM
hidden_dim = config.hidden_dim
output_dim = config.output_dim
LEARNING_RATE = config.LEARNING_RATE
decay_rate = config.decay_rate
beta = config.beta
batch_size = config.batch_size
EPOCHS = config.EPOCHS
activation = config.activation

# train model and save best model
model = setup_model(input_size=INPUT_DIM, hidden_size=hidden_dim, output_size=output_dim)
model, best_model, history_train_losses, history_train_accuracies, history_val_losses, history_val_accuracies = optimize_model_parameters(model, X_train, y_train, X_test, y_test, LEARNING_RATE, decay_rate, beta, EPOCHS, batch_size,activation)
store_model_parameters(best_model,"best_model")

# plot loss and accuracy
#fig, ax = plt.subplots(2, 1, figsize=(12, 8))
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(history_train_losses, label='Training loss')
ax1.plot(history_val_losses, label='Testing loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax2.plot(history_train_accuracies, label='Training accuracy')
ax2.plot(history_val_accuracies, label='Testing accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig('accuracy_and_loss.png')