# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 19:48
# @Author  : Zhang Jiaqi
# @File    : w3_2convolution.py
# @Description: C1W3 Implementing convolutional layers

import tensorflow as tf
import tensorflow.keras as keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy')
model.summary()
model.fit(train_images, train_labels, epochs=5)

test_loss = model.evaluate(test_images, test_labels)
print(f"loss of the test: {test_loss}")


