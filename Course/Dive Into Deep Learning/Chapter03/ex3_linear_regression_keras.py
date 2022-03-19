# -*- coding: utf-8 -*-
# @Time    : 2022/3/19 14:21
# @Author  : Zhang Jiaqi
# @File    : ex3_linear_regression_keras.py
# @Description:

import tensorflow as tf
import d2l.tensorflow as d2l

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个TensorFlow数据迭代器"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

def train(num_epochs, data_iter):
    initializer = tf.initializers.RandomNormal(stddev=0.01)
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))
    loss = tf.keras.losses.MeanSquaredError()
    trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

    for epoch in range(num_epochs):
        for X, y in data_iter:
            # GradientTape - 梯度流 记录磁带
            with tf.GradientTape() as tape:
                l = loss(net(X, training=True), y)
            grads = tape.gradient(l, net.trainable_variables)
            trainer.apply_gradients(zip(grads, net.trainable_variables))
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net.get_weights()[0]
    print('w的估计误差：', true_w - tf.reshape(w, true_w.shape))
    b = net.get_weights()[1]
    print('b的估计误差：', true_b - b)


if __name__ == "__main__":
    true_w = tf.constant([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)

    num_epochs = 3
    train(num_epochs, data_iter)





