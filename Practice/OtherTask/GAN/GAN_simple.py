# -*- coding: utf-8 -*-
# @Time    : 2022/4/20 16:40
# @Author  : Zhang Jiaqi
# @File    : GAN_simple.py
# @Description: 深入浅出GAN生成对抗网络
# 4.3 TensorFlow实现朴素GAN

import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def show_single_image(img_arr):
    """
    展示一张图片
    :param img_arr:
    :return:
    """
    # plt.imshow(img_arr, cmap="binary")
    plt.imshow(img_arr, cmap="Greys_r")
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
            plt.subplot(n_rows, n_cols, index + 1)  # subplot绘制子图
            plt.imshow(x_data[index], cmap="binary", interpolation="nearest")
            plt.axis('off')  # 坐标轴不可见
            # plt.title(class_names[y_data[index]])
    plt.show()

from tensorflow.examples.tutorials.mnist import input_data

def load_mnist_data():

    mnist = input_data.read_data_sets('./0data/MNIST_data')
    img = mnist.train.images[500]
    plt.imshow(img.reshape((28, 28)), cmap='Greys_r')
    plt.show()

    print(type(img))
    print(img.shape)

    return mnist

    # mnist = tf.keras.0datasets.mnist
    # (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()
    #
    # x_valid, x_train = x_train_all[:500], x_train_all[500:]
    # y_valid, y_train = y_train_all[:500], y_train_all[500:]
    #
    # print(f"x_valid.shape: {x_valid.shape}, y_valid.shape: {y_valid.shape}")
    # print(f"x_train.shape: {x_train.shape}, y_train.shape: {y_train.shape}")
    # print(f"x_test.shape: {x_test.shape}, y_test.shape: {y_test.shape}\n")
    #
    # # show_single_image(x_train[1])
    #
    # print(type(x_train[1]))
    # print(x_train.shape)
    #
    # # show_imgs(3, 5, x_train, y_train)

def get_inputs(real_size, noise_size):
    """
    接收输入 - 占位符来获得输入的数据
    :param real_size:
    :param noise_size:
    :return:
    """
    real_img = tf.placeholder(tf.float32, [None, real_size], name='real_img')
    noise_img = tf.placeholder(tf.float32, [None, noise_size], name='noise_img')

    return real_img, noise_img

def generator(noise_img, n_units, out_dim, reuse=False, alpha=0.01):
    """
    生成器
    :param noise_img: 生成器生成的噪声图片
    :param n_units: 隐藏层单元数
    :param out_dim: 生成器输出的tensor的size - 32x32=784
    :param reuse: 是否重用空间
    :param alpha: leaky ReLU系数
    :return:
    """
    # 创建一个名为generator的空间
    with tf.variable_scope("generator", reuse=reuse):
        # 输入层 - 隐藏层 - 输出层
        hidden1 = tf.layers.dense(noise_img, n_units)
        # 返回通过Leaky ReLU激活后较大的值
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        hidden1 = tf.layers.dropout(hidden1, rate=0.2, training=True)

        logits = tf.layers.dense(hidden1, out_dim)
        outputs = tf.tanh(logits)
        return logits, outputs


def discriminator(img, n_units, reuse=False, alpha=0.01):
    """
    判别器
    :param img: 图片
    :param n_units: 隐藏层单元数
    :param reuse: 是否重用空间
    :param alpha: leaky ReLU系数
    :return:
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        hidden1 = tf.layers.dense(img, n_units)
        hidden1 = tf.maximum(alpha * hidden1, hidden1)
        logits = tf.layers.dense(hidden1, 1)
        outputs = tf.sigmoid(logits)

        return logits, outputs


def train(mnist):
    # x_valid.shape: (500, 28, 28), y_valid.shape: (500,)
    # img_size = 28
    img_size = mnist.train.images[0].shape[0]
    # 噪声 Generator的初始输入
    noise_size = 100
    # 生成器隐藏层参数
    g_units = 128
    d_units = 128

    # leaky ReLU参数
    alpha = 0.01
    learning_rate = 0.001
    # 标签平滑
    smooth = 0.1
    # 重置default graph计算图以及nodes节点
    tf.reset_default_graph()

    real_img, noise_img = get_inputs(real_size=img_size, noise_size=noise_size)
    print(f"real_img: {real_img}")
    print(f"noise_img: {noise_img}")

    # 生成器
    g_logits, g_outputs = generator(noise_img, g_units, img_size)

    # 判别器
    d_logits_real, d_outputs_real = discriminator(real_img, d_units)

    # 传入生成图片，为其打分
    d_logits_fake, d_outputs_fake = discriminator(g_outputs, d_units, reuse=True)


    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_real, labels=tf.ones_like(d_logits_real)
    ) * (1 - smooth))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_fake, labels=tf.zeros_like(d_logits_fake)
    ))

    d_loss = tf.add(d_loss_real, d_loss_fake)
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=d_logits_fake, labels=tf.ones_like(d_outputs_fake)
    ) * (1 - smooth))

    train_vars = tf.trainable_variables()

    g_vars = [var for var in train_vars if var.name.startswith("generator")]
    d_vars = [var for var in train_vars if var.name.startswith("discriminator")]

    d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
    g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    batch_size = 64
    epochs = 10
    n_sample = 25
    samples = []
    losses = []
    saver = tf.train.Saver(var_list=g_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epochs):
            for batch_i in range(mnist.train.num_examples // batch_size):
                batch = mnist.train.next_batch(batch_size)
                batch_images = batch[0].reshape((batch_size, 784))
                batch_images = batch_images * 2 - 1
                batch_noise = np.random.uniform(-1, 1, size=(batch_size, noise_size))
                _ = sess.run(d_train_opt, feed_dict={
                    real_img: batch_images,
                    noise_img: batch_noise
                })
                _ = sess.run(g_train_opt, feed_dict={
                    noise_img: batch_noise
                })

                train_loss_d = sess.run(d_loss, feed_dict={
                    real_img: batch_images,
                    noise_img: batch_noise
                })

                train_loss_d_real = sess.run(d_loss_real, feed_dict={
                    real_img: batch_images,
                    noise_img: batch_noise
                })

                train_loss_d_fake = sess.run(d_loss_fake, feed_dict={
                    real_img: batch_images,
                    noise_img: batch_noise
                })

                train_loss_g = sess.run(g_loss, feed_dict={
                    noise_img: batch_noise
                })

                print("epoch {}/{}...".format(e + 1, epochs),
                      "total loss of D: {:.4f} (real_img_loss: {:.4f} + fake_img_loss: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
                      "total loss of G: {:.4f}".format(train_loss_g))

                losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

                sample_noise = np.random.uniform(-1, 1, size=(n_sample, noise_size))

                gen_samples = sess.run(generator(noise_img, g_units, img_size, reuse=True),
                                       feed_dict={noise_img: sample_noise})
                samples.append(gen_samples)
                saver.save(sess, './experiment/20220421/generator.ckpt')
                with open('./experiment/20220421/train_samples.pkl', 'wb') as f:
                    pickle.dump(samples, f)


def visualize():
    with open('./data/train_samples.pkl', 'rb') as f:
        samples = pickle.load(f)

    def view_img(epoch, samples):
        fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharex=True, sharey=True)
        for ax, img in zip(axes.flatten(), samples[epoch][1]):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
            plt.show()

    # view_img(-1, samples)

    def view_all():
        epoch_index = [x for x in range(0, 500, 50)]
        show_imgs = []
        for i in epoch_index:
            show_imgs.append(samples[i][1])

        rows, cols = len(epoch_index), len(samples[0][1])
        fig, axes = plt.subplots(figsize=(30, 20), nrows=rows, ncols=cols, sharex=True, sharey=True)
        index = range(0, 500, int(500/rows))
        for sample, ax_row in zip(show_imgs, axes):
            for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
                ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
        plt.show()

    view_all()



if __name__ == "__main__":
    mnist = load_mnist_data()
    train(mnist)

    # visualize()