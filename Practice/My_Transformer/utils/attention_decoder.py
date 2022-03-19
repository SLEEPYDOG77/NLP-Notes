# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 21:22
# @Author  : Zhang Jiaqi
# @File    : attention_decoder.py
# @Description:
from decoder import Decoder

class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口
    Defined in :numref:`sec_seq2seq_attention`"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError
