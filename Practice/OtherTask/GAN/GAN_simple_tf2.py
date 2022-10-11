# -*- coding: utf-8 -*-
# @Time    : 2022/4/23 14:08
# @Author  : Zhang Jiaqi
# @File    : GAN_simple_tf2.py
# @Description:
# dataset: http://yann.lecun.com/exdb/mnist/

import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt

def show_single_image(img_arr):
    """
    展示一张图片
    :param img_arr:
    :return:
    """
    plt.imshow(img_arr, cmap="binary")
    # plt.imshow(img_arr, cmap="Greys_r")
    plt.show()


def show_imgs(n_rows, n_cols, x_data, y_data):
    """
    展示多张图片
    :param n_rows:
    :param n_cols:
    :param x_data:
    :param y_data:
    :return:
    """
    assert len(x_data) == len(y_data)
    assert n_rows * n_cols < len(x_data)

    plt.figure(figsize=(n_cols * 1.4, n_rows * 1.6))

    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            # subplot绘制子图
            plt.subplot(n_rows, n_cols, index + 1)
            # plt.imshow(x_data[index], cmap="binary", interpolation="nearest")
            plt.imshow(x_data[index], cmap="Greys_r", interpolation="nearest")
            # 坐标轴不可见
            plt.axis('off')
            # plt.title(class_names[y_data[index]])
    plt.show()


def load_mnist_data():
    mnist = tf.keras.datasets.mnist
    (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()

    x_valid, x_train = x_train_all[:500], x_train_all[500:]
    y_valid, y_train = y_train_all[:500], y_train_all[500:]

    print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
    print(f"x_valid.shape: {x_valid.shape}, y_valid.shape: {y_valid.shape}")
    print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}\n")

    show_single_image(x_train[1])
    show_imgs(3, 5, x_train, y_train)

    return x_train, y_train, x_test, y_test


def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(f"x_train.shape 1: {x_train.shape}")
    x_train = (x_train.astype(np.float32) - 127.5)/127.5

    print(f"x_train.shape 2: {x_train.shape}")

    # 将图片转为向量 x_train from (60000, 28, 28) to (60000, 784) - 每一行 784 个元素
    x_train = x_train.reshape(60000, 784)
    print(f"x_train.shape 3: {x_train.shape}")
    return (x_train, y_train, x_test, y_test)


def create_generator():
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, input_dim=100),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(units=512),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(units=1024),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(units=784, activation='tanh'),
    ])

    generator.compile(
        loss='binary_crossentropy',
        optimizer=tf.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    )
    print("\nModel Generator: \n")
    generator.summary()
    return generator


def create_discriminator():
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1024, input_dim=784),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=512),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=256),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))
    print("\nModel Discriminator: \n")
    discriminator.summary()
    return discriminator


def create_gan(discriminator, generator):
    discriminator.trainable = False
    # 这是一个链式模型：输入经过生成器、判别器得到输出
    gan_input = tf.keras.Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = tf.keras.Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    print("\nModel GAN: \n")
    gan.summary()
    return gan


def plot_generated_images(epoch, generator, filepath, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{filepath}/gan_generated_image_{epoch}.png')


def train(epochs=1, batch_size=128):
    # 导入数据
    (x_train, y_train, x_test, y_test) = load_data()
    batch_count = x_train.shape[0] / batch_size

    # 定义生成器、判别器和GAN网络
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan(discriminator, generator)

    filepath = f'{os.getcwd()}/experiment/{time.strftime("%Y%m%d_%H%M%S", time.localtime())}'
    print(filepath)
    os.mkdir(filepath)

    for e in range(1, epochs + 1):
        print("\nEpoch %d" % e)
        for _ in tqdm(range(int(batch_count))):
            # 产生噪声喂给生成器
            noise = np.random.normal(0, 1, [batch_size, 100])

            # 产生假图片
            generated_images = generator.predict(noise)

            # 一组随机真图片
            image_batch = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]

            # 真假图片拼接
            X = np.concatenate([image_batch, generated_images])

            # 生成数据和真实数据的标签
            y_dis = np.zeros(2 * batch_size)
            y_dis[:batch_size] = 0.9

            # 预训练，判别器区分真假
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # 欺骗判别器 生成的图片为真的图片
            noise = np.random.normal(0, 1, [batch_size, 100])
            y_gen = np.ones(batch_size)

            # GAN的训练过程中判别器的权重需要固定
            discriminator.trainable = False

            # GAN的训练过程为交替“训练判别器”和“固定判别器权重训练链式模型”
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 5 == 0:
            # 画图 看一下生成器能生成什么
            plot_generated_images(e, generator, filepath)


if __name__ == "__main__":
    train(50, 256)
