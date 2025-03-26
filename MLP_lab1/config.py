# 参数查找环节要求调节学习率、隐藏层大小、正则化强度等超参数，观察并记录模型在不同超参数下的性能

class Hyperparameters:
    LEARNING_RATE = 0.01
    LR_DECAY = 0.95
    HIDDEN_SIZES = [128, 64]
    REG_STRENGTH = 1e-4
    EPOCHS = 50
    BATCH_SIZE = 128
    PRINT_EVERY = 5
    MODEL_SAVE_DIR = 'best_model'

def update_learning_rate(self):
        self.learning_rate *= self.lr_decay