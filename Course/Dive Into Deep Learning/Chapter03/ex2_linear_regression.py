# -*- coding: utf-8 -*-
# @Time    : 2022/3/18 20:36
# @Author  : Zhang Jiaqi
# @File    : ex2_linear_regression.py
# @Description:

import random
import tensorflow as tf
import d2l.tensorflow as d2l

# generate dataset
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y

def data_iter(batch_size, features, labels):
    """
    生成大小为 batch_size 的小批量
    :param batch_size: 批量大小
    :param features:  特征矩阵
    :param labels:  标签向量
    :return:
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)

def linreg(X, w, b):
    """
    线性回归模型
    :param X:
    :param w:
    :param b:
    :return:
    """
    return tf.matmul(X, w) + b

def squared_loss(y_hat, y):
    """
    均方误差损失函数
    :param y_hat:
    :param y:
    :return:
    """
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2

def sgd(params, grads, lr, batch_size):
    """
    小批量随机梯度下降
    :param params:
    :param grads:
    :param lr:
    :param batch_size:
    :return:
    """
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)


if __name__ == "__main__":
    lr = 0.03
    num_epochs = 3

    net = linreg
    loss = squared_loss
    batch_size = 10

    true_w = tf.constant([2, -3.4])
    true_b = 4.2

    W = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
    b = tf.Variable(tf.zeros(1), trainable=True)
    features, labels = synthetic_data(true_w, true_b, 1000)

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            with tf.GradientTape() as g:
                l = loss(net(X, W, b), y)
            dw, db = g.gradient(l, [W, b])
            sgd([W, b], [dw, db], lr, batch_size)
        train_l = loss(net(features, W, b), labels)
        print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

    print(f'w的估计误差: {true_w - tf.reshape(W, true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')
