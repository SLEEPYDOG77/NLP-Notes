# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 15:54
# @Author  : Zhang Jiaqi
# @File    : ex3_language_model.py
# @Description:

import random
from d2l import read_time_machine
from d2l import tokenize, Vocab
import torch

# 8.3.3 自然语言统计
def get_tokens():
    tokens = tokenize(read_time_machine())
    corpus = [token for line in tokens for token in line]
    vocab = Vocab(corpus)
    print(vocab.token_freqs[:10])

    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    trigram_vocab = Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

# 8.3.4 读取长序列数据
def seq_data_iter_random(corpus, batch_size, num_steps):
    """
    使用随机抽样生成一个小批量子序列
    :param corpus:
    :param batch_size:
    :param num_steps:
    :return:
    """
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)


if __name__ == '__main__':
    my_seq = list(range(35))
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)



