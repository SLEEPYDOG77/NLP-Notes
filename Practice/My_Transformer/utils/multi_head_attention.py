# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 21:19
# @Author  : Zhang Jiaqi
# @File    : multi_head_attention.py
# @Description:
import tensorflow as tf
from dot_product_attention import DotProductAttention
from compute_utils import transpose_qkv, transpose_output

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias)
        self.W_o = tf.keras.layers.Dense(num_hiddens, use_bias=bias)

    def call(self, queries, keys, values, valid_lens, **kwargs):
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)

        output = self.attention(queries, keys, values, valid_lens, **kwargs)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
    pass
