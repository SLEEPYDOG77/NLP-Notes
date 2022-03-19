# -*- coding: utf-8 -*-
# @Time    : 2022/3/9 21:15
# @Author  : Zhang Jiaqi
# @File    : encoder_decoder.py
# @Description:

import tensorflow as tf

class EncoderDecoder(tf.keras.Model):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args, **kwargs):
        enc_outputs = self.encoder(enc_X, *args, **kwargs)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state, **kwargs)
