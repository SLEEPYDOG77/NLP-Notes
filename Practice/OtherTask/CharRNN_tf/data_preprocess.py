# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 11:27
# @Author  : Zhang Jiaqi
# @File    : data_preprocess.py
# @Description:

# from pprint import pprint
# from utils.vocab import Vocabulary
#
# def load_data(data_dir, output_dir):
#     with open(data_dir, 'r', encoding='utf-8') as f:
#         text_file = f.readlines()
#
#     word_list = [v for s in text_file for v in s]
#     Vocabulary(word_list, max_vocab=50000)
#
#
# if __name__ == "__main__":
#     data_dir = "../0data/chinese_poetry_data/train.txt"
#     output_dir = "../0data/chinese_poetry_data/"
#
#     load_data(data_dir, output_dir)
#     pass