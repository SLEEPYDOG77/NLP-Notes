# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 21:16
# @Author  : Zhang Jiaqi
# @File    : decoder.py
# @Description:

import tensorflow as tf

class Decoder(tf.keras.layers.Layer):
    """编码器-解码器架构的基本解码器接口
    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        # 将编码器的输出（enc_outputs）转换为编码后的状态
        raise NotImplementedError

    def call(self, X, state, **kwargs):
        raise NotImplementedError

