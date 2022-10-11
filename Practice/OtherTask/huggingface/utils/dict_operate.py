# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 19:36
# @Author  : Zhang Jiaqi
# @File    : dict_operate.py
# @Description:


class dict_operate(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_vocab(self):
        my_dict = self.tokenizer.get_vocab()
        return my_dict

    def add_new_word(self, new_words):
        self.tokenizer.add_tokens(new_words=[new_word for new_word in new_words])

    def add_new_tokens(self, token_name, new_token):
        self.tokenizer.add_special_tokens({token_name, new_token})

