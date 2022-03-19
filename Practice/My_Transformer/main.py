# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 18:48
# @Author  : Zhang Jiaqi
# @File    : main.py
# @Description:
import collections
import os
import tensorflow as tf
from utils.transformer_encoder import TransformerEncoder
from utils.transformer_decoder import TransformerDecoder
from utils.encoder_decoder import EncoderDecoder

def try_gpu(i=0):
    if len(tf.config.experimental.list_logical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def read_data():
    with open(os.path.join('data', 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def preprocess(raw_text):
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    text = raw_text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = []
    for i, char in enumerate(text):
        if i > 0 and no_space(char, text[i - 1]):
            out.append(' ' + char)
        else:
            out.append(char)
    return ''.join(out)

def tokenize(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        new_tokens = []
        for line in tokens:
            for token in line:
                new_tokens.append(token)
    return collections.Counter(new_tokens)

class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)

        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def get_corpus(tokens, vocab):
    corpus = []
    for line in tokens:
        for token in line:
            corpus.append(vocab[token])
    return corpus

def truncate_pad(line, num_steps, padding_token):
    # 用 <pad> 填充或截断
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def build_array(lines, vocab, num_steps):
    # 把每一行文本序列转换为数字序列
    lines = [vocab[l] for l in lines]
    # 在每一行数字序列后添加 <eos>
    lines = [l + [vocab['<eos>']] for l in lines]

    # 构建array
    array = tf.constant([truncate_pad(l, num_steps, vocab['<pad>']) for l in lines])
    # 获取每一行的有效长度 排除了填充词元
    valid_len = tf.reduce_sum(tf.cast(array != vocab['<pad>'], tf.int32), axis=1)
    return array, valid_len

def load_array(data_arrays, batch_size, is_train=True):
    # 把给定的元祖、列表和张量等数据进行特征切片
    # 切片的范围是从最外层维度开始的
    # 如果有多个特征进行组合，那么一次切片是把每个组合的最外维度的数据切开，分成一组一组的
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    pass


if __name__ == '__main__':
    raw_text = read_data()
    # 文本预处理
    text = preprocess(raw_text)
    # 词元化
    source, target = tokenize(text)
    # 字典
    src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 语料
    src_corpus = get_corpus(source, src_vocab)
    tgt_corpus = get_corpus(target, tgt_vocab)

    num_steps = 10
    src_array, src_valid_len = build_array(source, src_vocab, num_steps=10)
    tgt_array, tgt_valid_len = build_array(target, tgt_vocab, num_steps=10)

    batch_size = 64
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    train_iter = load_array(data_arrays, batch_size)

    num_hiddens = 32
    num_layers = 2
    dropout = 0.1
    lr = 0.005
    num_ephochs = 200
    device = try_gpu()
    ffn_num_input = 32
    ffn_num_hiddens = 64
    num_heads = 4
    key_size = 32
    query_size = 32
    value_size = 32
    norm_shape = [2]

    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
        ffn_num_hiddens, num_heads, num_layers, dropout
    )
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens, norm_shape,
        ffn_num_hiddens, num_heads, num_layers, dropout)

    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_ephochs, tgt_vocab, device)

    # print('\npredict--------------------')
    # engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    # fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    # for eng, fra in zip(engs, fras):
    #     translation, dec_attention_weight_seq = predict_seq2seq(
    #         net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    #     print(f'{eng} => {translation}, ',
    #           f'bleu {bleu(translation, fra, k=2):.3f}')
    #
    # pass