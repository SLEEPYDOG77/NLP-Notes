# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 14:52
# @Author  : Zhang Jiaqi
# @File    : ex6_softmax.py
# @Description: 3.6 softmax回归的从零开始实现

import tensorflow as tf
import d2l.tensorflow
from utils.Accumulator import Accumulator
from utils.Animator import Animator
from utils.Updater import Updater

# initial parameters
# 28 x 28
num_inputs = 784
# 10个类别
num_outputs = 10

# stddev 标准偏差
W = tf.Variable(tf.random.normal(shape=(num_inputs, num_outputs), mean=0, stddev=0.01))
b = tf.Variable(tf.zeros(num_outputs))

def load_data():
    batch_size = 256
    train_iter, test_iter = d2l.tensorflow.load_data_fashion_mnist(batch_size)
    return train_iter, test_iter

def softmax(X):
    # 1. 对每个项求幂
    X_exp = tf.exp(X)

    # 2. 对每一行求和，得到每个样本的规范化常数
    partition = tf.reduce_sum(X_exp, 1, keepdims=True)

    # 3. 将每一行除以其规范化常数，确保结果的和为1
    return X_exp / partition

def net(X):
    return softmax(tf.matmul(tf.reshape(X, (-1, W.shape[0])), W) + b)

def cross_entropy(y_hat, y):
    return -tf.math.log(tf.boolean_mask(y_hat, tf.one_hot(y, depth=y_hat.shape[-1])))

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = tf.argmax(y_hat, axis=1)
    cmp = tf.cast(y_hat, y.dtype) == y
    return float(tf.reduce_sum(tf.cast(cmp, y.dtype)))

def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.tensorflow.size(y))
    return metric[0] / metric[1]

def train_epoch(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第3章）"""
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        with tf.GradientTape() as tape:
            y_hat = net(X)
            # Keras内置的损失接受的是（标签，预测），这不同于用户在本书中的实现。
            # 本书的实现接受（预测，标签），例如我们上面实现的“交叉熵”
            if isinstance(loss, tf.keras.losses.Loss):
                l = loss(y, y_hat)
            else:
                l = loss(y_hat, y)
        if isinstance(updater, tf.keras.optimizers.Optimizer):
            params = net.trainable_variables
            grads = tape.gradient(l, params)
            updater.apply_gradients(zip(grads, params))
        else:
            updater(X.shape[0], tape.gradient(l, updater.params))
        # Keras的loss默认返回一个批量的平均损失
        l_sum = l * float(tf.size(y)) if isinstance(
            loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l_sum, accuracy(y_hat, y), tf.size(y))
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.tensorflow.get_fashion_mnist_labels(y)
    preds = d2l.tensorflow.get_fashion_mnist_labels(tf.argmax(net(X), axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.tensorflow.show_images(
        tf.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])

if __name__ == "__main__":
    train_iter, test_iter = load_data()
    updater = Updater([W, b], lr=0.1)
    num_epochs = 10
    train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    # net.save_weights('my_image_classifier')
    predict(net, test_iter)

