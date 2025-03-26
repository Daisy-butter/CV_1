# 参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能

from train import Trainer
from MLP_model import MLP

lr = 0.01
hidden_sizes = [64, 32]
reg_strength = 1e-3
epochs = 20
batch_size = 128
print_every = 5
