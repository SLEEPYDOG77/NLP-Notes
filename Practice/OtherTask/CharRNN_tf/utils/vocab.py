# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 13:36
# @Author  : Zhang Jiaqi
# @File    : vocab.py
# @Description:

from pprint import pprint

class Vocabulary(object):
    def __init__(self, word_list=None, max_vocab=None):
        self.vocab = set(word_list)

        vocab_count = {}
        for word in self.vocab:
            vocab_count[word] = 0
        for word in word_list:
            vocab_count[word] += 1

        vocab_count_list = []
        for word in vocab_count:
            vocab_count_list.append((word, vocab_count[word]))
        vocab_count_list.sort(key=lambda x: x[1], reverse=True)
        if len(vocab_count_list) > max_vocab:
            vocab_count_list = vocab_count_list[:max_vocab]
        vocab = [x[0] for x in vocab_count_list]
        self.vocab = vocab
        self.word2int_table = {c: i for i, c in enumerate(self.vocab)}
        self.int2word_table = dict(enumerate(self.vocab))

    def word2int(self, word):
        return self.word2int_table[word]

    def int2word(self, int_word):
        return self.int2word_table[int_word]



