# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 21:20
# @Author  : Zhang Jiaqi
# @File    : add_norm.py
# @Description:

import tensorflow as tf

class AddNorm(tf.keras.layers.Layer):
    """残差连接后进行层规范化
    Defined in :numref:`sec_transformer`"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(normalized_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)
