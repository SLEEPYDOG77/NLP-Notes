# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 15:01
# @Author  : Zhang Jiaqi
# @File    : ex2_mlp.py
# @Description:

import tensorflow as tf
import d2l.tensorflow as d2l

def relu(X):
    return tf.math.maximum(X, 0)

def loss(y_hat, y):
    return tf.losses.sparse_categorical_crossentropy(
        y, y_hat, from_logits=True)

def train(X):
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    W1 = tf.Variable(tf.random.normal(
        shape=(num_inputs, num_hiddens), mean=0, stddev=0.01))
    b1 = tf.Variable(tf.zeros(num_hiddens))
    W2 = tf.Variable(tf.random.normal(
        shape=(num_hiddens, num_outputs), mean=0, stddev=0.01))
    b2 = tf.Variable(tf.zeros(num_outputs))

    params = [W1, b1, W2, b2]

    X = tf.reshape(X, (-1, num_inputs))
    H = relu(tf.matmul(X, W1) + b1)
    return tf.matmul(H, W2) + b2



if __name__ == "__main__":
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)



