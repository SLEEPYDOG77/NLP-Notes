# -*- coding: utf-8 -*-
# @Time    : 2022/3/25 19:35
# @Author  : Zhang Jiaqi
# @File    : train.py
# @Description:
import json
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf

min_count = 32
maxlen = 400
batch_size = 64
epochs = 100
char_size = 128
z_dim = 128

data_dir = 'data/lcsts_data.json'

if os.path.exists('seq2seq_config.json'):
    chars, id2char, char2id = json.load(open('seq2seq_config.json'))
    id2char = {
        int(i): j for i, j in id2char.items()
    }
else:
    chars = {}
    with open(data_dir, "r", encoding="utf-8") as f:
        source = json.load(data_dir)
        for item in tqdm(source):
            for w in item['title']:
                chars[w] = chars.get(w, 0) + 1
            for w in item['content']:
                chars[w] = chars.get(w, 0) + 1
        chars = {
            i: j for i, j in chars.items() if j >= min_count
        }

        id2char = {
            i + 4: j for i, j in enumerate(chars)
        }

        char2id = {
            j: i for i, j in id2char.items()
        }

        json.dump([chars, id2char, char2id], open('seq2seq_config.json', 'w'))


def str2id(s, start_end=False):
    # 文字转整数id
    if start_end:  # 补上<start>和<end>标记
        ids = [char2id.get(c, 1) for c in s[:maxlen - 2]]
        ids = [2] + ids + [3]
    else:  # 普通转化
        ids = [char2id.get(c, 1) for c in s[:maxlen]]
    return

def id2str(ids):
    # id转文字，找不到的用空字符代替
    return ''.join([id2char.get(i, '') for i in ids])


def padding(x):
    ml = max([len(i) for i in x])
    return [i + [0] * (ml - len(i)) for i in x]


def data_generator(data_dir):
    x, y = [], []
    with open(data_dir, "r", encoding="utf-8") as f:
        source = json.load(data_dir)
        for item in source:
            x.append(str2id(item['content']))
            y.append(str2id(item['title'], start_end=True))
            if len(x) == batch_size:
                x = np.array(padding(x))
                y = np.array(padding(y))
                yield [x, y], None

def to_one_hot(x):
    x, x_mask = x
    x = tf.keras.backend.cast(x, 'int32')
    x = tf.keras.backend.one_hot(x, len(chars) + 4)
    x = tf.keras.backend.sum(x_mask * x, 1, keepdims=True)
    x = tf.keras.backend.cast(tf.keras.backend.greater(x, 0.5), 'float32')
    return x