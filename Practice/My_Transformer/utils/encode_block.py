# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 21:20
# @Author  : Zhang Jiaqi
# @File    : encode_block.py
# @Description:
import tensorflow as tf
from multi_head_attention import MultiHeadAttention
from add_norm import AddNorm
from position_wise_ffn import PositionWiseFFN

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                            num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs), **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)
