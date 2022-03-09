# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 21:16
# @Author  : Zhang Jiaqi
# @File    : encoder.py
# @Description:

import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def call(self, X, *args, **kwargs):
        raise NotImplementedError

