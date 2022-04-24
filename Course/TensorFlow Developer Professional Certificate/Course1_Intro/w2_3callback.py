# -*- coding: utf-8 -*-
# @Time    : 2022/4/24 19:27
# @Author  : Zhang Jiaqi
# @File    : w2_3callback.py
# @Description: C1W2 Using Callbacks to control training

import tensorflow as tf
import tensorflow.keras as keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('loss') < 0.4):
            print("\nLoss is low so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()
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
model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])

eva = model.evaluate(test_images, test_labels)
print(f"loss of the evaluation: {eva}")
