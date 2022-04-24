# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 19:00
# @Author  : Zhang Jiaqi
# @File    : w2_2load_data.py
# @Description: C1W2 Writing code to load training data

import tensorflow as tf
import tensorflow.keras as keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# model.compile(optimizer=tf.train.AdamOptimizer(), loss='sparse_categorical_crossentropy')
model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')
model.fit(train_images, train_labels, epochs=5)

eva = model.evaluate(test_images, test_labels)
print(f"loss of the evaluation: {eva}")
