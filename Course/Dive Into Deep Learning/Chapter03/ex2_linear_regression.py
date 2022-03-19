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


if __name__ == "__main__":
    true_w = tf.constant([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    print(len(features))
    print(len(labels))

    d2l.set_figsize()
    d2l.plt.scatter(features[:, (0)].numpy(), labels.numpy(), 1)
    d2l.plt.show()

