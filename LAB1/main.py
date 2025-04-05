from MLP_model import (
    setup_model,
    model_forward_with_bn,
    model_backward_with_bn,
    update_parameters,
    calculate_loss,
    calculate_accuracy,
    store_model_parameters
)
from config import config
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# 数据路径
data_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/计算机视觉/lab_data"

X_train = np.load(f"{data_dir}/X_train.npy")
y_train = np.load(f"{data_dir}/y_train.npy")
X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

# 超参数
INPUT_DIM = config.INPUT_DIM
hidden_dim = config.hidden_dim
output_dim = config.output_dim
LEARNING_RATE = config.LEARNING_RATE
decay_rate = config.decay_rate
beta = config.beta
batch_size = config.batch_size
EPOCHS = config.EPOCHS
activation = config.activation

# 初始化模型
model = setup_model(INPUT_DIM, hidden_dim, output_dim)

# 训练
history_train_losses = []
history_train_accuracies = []
history_val_losses = []
history_val_accuracies = []
best_model = None
best_val_accuracy = 0

for epoch in range(1, EPOCHS + 1):
    # 动态调整学习率
    learn_rate = LEARNING_RATE / (1 + decay_rate * epoch)

    # 分批训练
    num_batches = X_train.shape[0] // batch_size
    train_loss = 0
    train_correct = 0

    for batch in range(num_batches):
        batch_start = batch * batch_size
        batch_end = batch_start + batch_size
        X_batch = X_train[batch_start:batch_end]
        y_batch = y_train[batch_start:batch_end]

        # 前向传播
        y_hat, activations, caches = model_forward_with_bn(model, X_batch, activation)

        # 计算损失
        batch_loss = calculate_loss(y_batch, y_hat, model, beta)
        train_loss += batch_loss

        # 反向传播
        gradients = model_backward_with_bn(model, activations, caches, X_batch, y_batch, y_hat, beta, activation)
        model = update_parameters(model, gradients, learn_rate)

        # 计算准确率
        batch_predictions = np.argmax(y_hat, axis=1)
        batch_labels = np.argmax(y_batch, axis=1)
        train_correct += np.sum(batch_predictions == batch_labels)

    # 记录训练指标
    train_loss /= num_batches
    train_accuracy = train_correct / len(y_train)
    history_train_losses.append(train_loss)
    history_train_accuracies.append(train_accuracy * 100)

    # 验证
    y_hat_val, _, _ = model_forward_with_bn(model, X_test, activation)
    val_loss = calculate_loss(y_test, y_hat_val, model, beta)
    val_accuracy = calculate_accuracy(y_hat_val, y_test) * 100

    history_val_losses.append(val_loss)
    history_val_accuracies.append(val_accuracy)

    # 打印当前 epoch 信息
    print(f"Epoch {epoch}/{EPOCHS}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy * 100:.4f}%, "
          f"Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}%")

    # 保存最佳模型
    if val_accuracy > best_val_accuracy:
        best_model = model.copy()
        best_val_accuracy = val_accuracy

# 保存最佳模型
store_model_parameters(best_model, "best_model")

# 绘制训练曲线
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.plot(history_train_losses, label="Training Loss")
ax1.plot(history_val_losses, label="Validation Loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(history_train_accuracies, label="Training Accuracy")
ax2.plot(history_val_accuracies, label="Validation Accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig("accuracy_and_loss.png")
plt.show()
