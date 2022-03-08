# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 14:29
# @Author  : Zhang Jiaqi
# @File    : transformer_translation_tf.py
# @Description: 基于Transformer的英语-法语翻译 - tensorflow

import os.path
import tensorflow as tf
from d2l import DATA_URL, DATA_HUB, download_extract
from d2l import Vocab
from d2l import try_gpu
from d2l import load_array
from d2l import reduce_sum, astype, int32
from utils_tf.transformer_decoder_tf import TransformerDecoder
from utils_tf.transformer_encoder_tf import TransformerEncoder
from d2l import bleu
from utils_tf.seq2seq_tf import train_seq2seq, predict_seq2seq

# 下载英 - 法数据集
def read_data_nmt():
    DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')
    data_dir = download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

# 数据集预处理
def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # (其实就是在英文写作的时候，我们写的一些词组为了避免他们分开在两行导致人们阅读的时候看不懂，就要把它们写在一起，就用到了不间断空格。)
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


# 词元化 tokenize
def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

# 构建词表 - Vocab
# 将字符串类型的词元映射到从0开始的数字索引中
# 根据每个唯一词元的出现频率，为其分配一个数字索引

# 截断或填充文本序列
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

#@save
def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = tf.constant([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = tf.reduce_sum(
        tf.cast(array != vocab['<pad>'], tf.int32), 1)
    return array, valid_len

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

if __name__ == "__main__":
    print('read_data_nmt---------------')
    raw_text = read_data_nmt()
    print(raw_text[:75])

    print('preprocess_nmt--------------')
    text = preprocess_nmt(raw_text)
    print(text[:80])

    print('\ntokenize_nmt----------------')
    source, target = tokenize_nmt(text)
    print(source[:6])
    print(target[:6])

    print('\nsource vocab----------------')
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(list(src_vocab.token_to_idx.items())[:10])
    src_corpus = [src_vocab[token] for line in source for token in line]
    print(len(src_corpus), len(src_vocab))

    print('\ntarget vocab----------------')
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    print(list(tgt_vocab.token_to_idx.items())[:10])
    tgt_corpus = [tgt_vocab[token] for line in target for token in line]
    print(len(tgt_corpus), len(tgt_vocab))

    print('\ntraining--------------------')
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_ephochs, device = 0.005, 200, try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [2]

    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)

    train_iter = data_iter

    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
        ffn_num_hiddens, num_heads, num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
        ffn_num_hiddens, num_heads, num_layers, dropout)

    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_ephochs, tgt_vocab, device)

    print('\npredict--------------------')
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
              f'bleu {bleu(translation, fra, k=2):.3f}')





