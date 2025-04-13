import numpy as np
import json
from config import config

# Model parameter handling
def store_model_parameters(model, filename):
    model_data = {key: value.tolist() for key, value in model.items()}
    with open(f'{filename}.json', 'w') as file:
        json.dump(model_data, file)

def retrieve_model_parameters(filename):
    with open(f'{filename}.json', 'r') as file:
        model_data = json.load(file)
    return {key: np.array(value) for key, value in model_data.items()}

# Activation functions and their derivatives
def relu_activation(x):
    return np.maximum(x, 0)

def leaky_relu_activation(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def derivative_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax_activation(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# Loss computation
def calculate_loss(true_y, predicted_y, model, regularization_factor):
    reg_loss = regularization_factor / 2 * sum(np.sum(weights ** 2) for weights in [model['W1'], model['W2'], model['W3']])
    log_loss = -np.mean(np.sum(true_y * np.log(predicted_y + 1e-12), axis=1))
    return log_loss + reg_loss

def calculate_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))

# Batch Normalization forward pass
def batch_normalization_forward(x, gamma, beta, epsilon=1e-5):
    # 计算输入x的均值
    batch_mean = np.mean(x, axis=0)
    # 计算输入x的方差
    batch_var = np.var(x, axis=0)
    # 对输入x进行标准化
    x_normalized = (x - batch_mean) / np.sqrt(batch_var + epsilon)
    # 对标准化后的x进行缩放和平移
    out = gamma * x_normalized + beta
    # 保存中间变量，用于反向传播
    cache = (x, x_normalized, batch_mean, batch_var, gamma, beta, epsilon)
    return out, cache

def attention_forward(query, key, value, dropout_rate=0.01, training=True):
    # 提取 embedding 的维度
    embedding_dim = query.shape[-1]
    scale_factor = 1  # 控制注意力权重的温度
    
    # 点积注意力计算
    scores = query @ key.T / (np.sqrt(embedding_dim) * scale_factor)
    
    # Softmax归一化（计算注意力权重）
    attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # 数值稳定
    attention_weights /= np.sum(attention_weights, axis=1, keepdims=True)
    
    # 放宽权重裁剪限制
    attention_weights = np.clip(attention_weights, 1e-5, 1.0 - 1e-5)  # 权重裁剪范围调整为1e-5以减少信息丢失
    
    # 加权求和（注意力输出）
    attention_out = attention_weights @ value
    
    # Dropout操作（如果处于训练模式）
    if training and dropout_rate > 0:
        dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=attention_out.shape) / (1 - dropout_rate)
        attention_out *= dropout_mask

    return attention_out, attention_weights


# Batch Normalization backward pass
def batch_normalization_backward(dout, cache):
    # 获取缓存中的参数
    x, x_normalized, batch_mean, batch_var, gamma, beta, epsilon = cache
    # 获取输入数据的样本数
    m = x.shape[0]

    # 计算gamma的梯度
    dgamma = np.sum(dout * x_normalized, axis=0)
    # 计算beta的梯度
    dbeta = np.sum(dout, axis=0)

    # 计算x_normalized的梯度
    dx_normalized = dout * gamma
    # 计算batch_var的梯度
    dvar = np.sum(dx_normalized * (x - batch_mean) * -0.5 * (batch_var + epsilon)**(-1.5), axis=0)
    # 计算batch_mean的梯度
    dmean = np.sum(dx_normalized * -1 / np.sqrt(batch_var + epsilon), axis=0) + dvar * np.sum(-2 * (x - batch_mean), axis=0) / m
    # 计算x的梯度
    dx = dx_normalized / np.sqrt(batch_var + epsilon) + dvar * 2 * (x - batch_mean) / m + dmean / m

    # 返回x的梯度、gamma的梯度、beta的梯度
    return dx, dgamma, dbeta

# Dropout implementation
def dropout(x, rate, training=True):
    if training:
        mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
        return x * mask
    return x

# Setup the model with Batch Normalization and Dropout parameters
def setup_model(input_size, hidden_size, output_size):
    model = {
        'W1': np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size), # He initialization
        'b1': np.zeros(hidden_size),
        'gamma1': np.ones(hidden_size),
        'beta1': np.zeros(hidden_size),
        'W2': np.random.randn(hidden_size, hidden_size) * np.sqrt(2. / hidden_size),
        'b2': np.zeros(hidden_size),
        'gamma2': np.ones(hidden_size),
        'beta2': np.zeros(hidden_size),
        'W3': np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size),
        'b3': np.zeros(output_size),
    }
    return model

# Forward propagation with Batch Normalization and Dropout
def model_backward_with_bn_attention(
    model, activations, caches, X, y, y_hat, reg_strength, activation="leaky_relu"
):
    gradients = {}  # 储存梯度
    W1, W2, W3 = model['W1'], model['W2'], model['W3']
    gamma1, gamma2 = model['gamma1'], model['gamma2']

    # 第三层（输出层）
    error = y_hat - y
    gradients['dW3'] = activations['a2_attention'].T @ error + reg_strength * W3
    gradients['db3'] = np.sum(error, axis=0)

    # 第二层（Attention反向传播 + 残差连接）
    da2_attention_residual = error @ W3.T  # 从输出层反传来的梯度
    da2_attention = da2_attention_residual  # 注意力输出的部分梯度
    da2_activation_residual = da2_attention_residual  # 残差，直接传递梯度到激活层

    # 注意力权重梯度
    attention_weights = activations['attention_weights']
    dweights = da2_attention @ attention_weights.T  # 基于注意力反传
    dz2_attention, dgamma2, dbeta2 = batch_normalization_backward(dweights, caches['bn2'])

    # 第二层激活到残差部分的梯度
    da2_activation = dz2_attention + da2_activation_residual  # 激活层部分梯度（加残差）

    # 第一层（隐藏层）
    da1 = da2_activation @ W2.T
    if activation == 'leaky_relu':
        da1 *= derivative_leaky_relu(activations['a1'])
    dz1, dgamma1, dbeta1 = batch_normalization_backward(da1, caches['bn1'])
    gradients['dW1'] = X.T @ dz1 + reg_strength * W1
    gradients['db1'] = np.sum(dz1, axis=0)
    gradients['dgamma1'], gradients['dbeta1'] = dgamma1, dbeta1

    # 更新第二层权重
    gradients['dW2'] = activations['a1'].T @ dz2_attention + reg_strength * W2
    gradients['db2'] = np.sum(dz2_attention, axis=0)
    gradients['dgamma2'], gradients['dbeta2'] = dgamma2, dbeta2

    return gradients

def model_forward_with_bn_attention(
    model, X, activation="leaky_relu", dropout_rate=0.01, training=True, batch_size=config.batch_size
):
    num_samples = X.shape[0]
    activations = {}
    caches = {}

    # 参数解包
    W1, b1, gamma1, beta1 = model['W1'], model['b1'], model['gamma1'], model['beta1']
    W2, b2, gamma2, beta2 = model['W2'], model['b2'], model['gamma2'], model['beta2']
    W3, b3 = model['W3'], model['b3']

    # 初始化保存最终结果的数组
    predictions = np.zeros((num_samples, W3.shape[1]))  # 保存最终预测值

    # 迭代 mini-batch
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        X_batch = X[start_idx:end_idx]  # 当前 mini-batch 样本
        batch_size_actual = X_batch.shape[0]  # 实际 batch 大小

        # --- 第一层 ---
        z1 = X_batch @ W1 + b1
        a1_bn, bn_cache1 = batch_normalization_forward(z1, gamma1, beta1)
        a1 = leaky_relu_activation(a1_bn) if activation == 'leaky_relu' else relu_activation(a1_bn)
        a1 = dropout(a1, dropout_rate, training)
        activations['a1'], caches['bn1'] = a1, bn_cache1

        # --- 第二层 ---
        z2 = a1 @ W2 + b2
        a2_bn, bn_cache2 = batch_normalization_forward(z2, gamma2, beta2)
        a2_activation = (
            leaky_relu_activation(a2_bn) if activation == 'leaky_relu' else relu_activation(a2_bn)
        )
        caches['bn2'] = bn_cache2

        # --- Attention机制 ---
        query, key, value = a2_activation, a2_activation, a2_activation  # 注意力输入
        a2_attention, attention_weights = attention_forward(
            query=query,
            key=key,
            value=value,
            dropout_rate=dropout_rate,
            training=training
        )

        # 残差连接
        a2_attention_residual = a2_attention + a2_activation  # 注意力层输出加上激活层输入 (a2_activation)

        # 存储注意力结果和权重
        activations['a2_attention'] = a2_attention_residual  # 残差后注意力
        activations['attention_weights'] = attention_weights

        # --- 第三层（输出层） ---
        z3 = a2_attention_residual @ W3 + b3
        a3 = softmax_activation(z3)  # 当前 mini-batch 的预测
        predictions[start_idx:end_idx] = a3  # 保存到最终结果

    return predictions, activations, caches


def update_parameters(model, gradients, learn_rate):
    for key in model:
        model[key] -= learn_rate * gradients.get(f'd{key}', 0)
    return model
