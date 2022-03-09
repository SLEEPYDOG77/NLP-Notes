# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 21:20
# @Author  : Zhang Jiaqi
# @File    : position_wise_ffn.py
# @Description:

import tensorflow as tf

class PositionWiseFFN(tf.keras.layers.Layer):
    """基于位置的前馈网络
    Defined in :numref:`sec_transformer`"""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, **kwargs):
        super().__init__(*kwargs)
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))
