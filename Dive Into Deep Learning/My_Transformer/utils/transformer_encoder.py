# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 21:21
# @Author  : Zhang Jiaqi
# @File    : transformer_encoder.py
# @Description:
import tensorflow as tf
from encoder import Encoder
from positional_encoding import PositionalEncoding
from encode_block import EncoderBlock

class TransformerEncoder(Encoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_layers, dropout, bias=False, **kwargs):
        """
        :param vocab_size:
        :param key_size:
        :param query_size:
        :param value_size:
        :param num_hiddens:
        :param norm_shape:
        :param ffn_num_hiddens:
        :param num_heads:
        :param num_layers:
        :param dropout:
        :param bias:
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = [EncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias
        ) for _ in range(num_layers)]

    def call(self, X, valid_lens, **kwargs):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
