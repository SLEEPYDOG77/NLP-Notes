# -*- coding: utf-8 -*-
# @Time    : 2022/3/15 19:45
# @Author  : Zhang Jiaqi
# @File    : ex5_fashion_mnist.py
# @Description: 3.5 图像分类数据集

import tensorflow as tf
import matplotlib.pyplot as plt
from d2l.tensorflow import Timer

tf.use_svg_display()

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    print([text_labels[int(i)] for i in labels])


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.numpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def load_data(batch_size, resize=None):
    # load dataset
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    print(len(mnist_train))
    print(len(mnist_test))
    print(mnist_train[0][0].shape)

    # 归一化
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))

    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y
    )

    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn)
    )

if __name__ == "__main__":
    train_iter, test_iter = load_data(batch_size=32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break

