# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 19:39
# @Author  : Zhang Jiaqi
# @File    : ScaleShift.py
# @Description:

import tensorflow as tf
from tensorflow.keras.layers import *

class ScaleShift(Layer):
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def build(self, input_shape):
        kernel_shape = (1, ) * (len(input_shape) - 1) + (input_shape[-1])
        self.log_scale = self.add_weight(name='log_scale',
                                         shape=kernel_shape,
                                         initializer='zeros')
        self.shift = self.add_weight(name='shift',
                                     shape=kernel_shape,
                                     initializer='zeros')

    def call(self, inputs):
        x_outs = tf.keras.backend.exp(self.log_scale) * inputs + self.shift

        return x_outs

