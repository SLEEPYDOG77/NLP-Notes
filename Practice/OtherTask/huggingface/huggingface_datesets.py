# -*- coding: utf-8 -*-
# @Time    : 2022/3/31 20:07
# @Author  : Zhang Jiaqi
# @File    : test0_datesets.py
# @Description:
import json

from datasets import load_dataset
from tqdm import tqdm

dataset = load_dataset('gigaword')
print(dataset)
print(len(dataset['validation']))

train_lst = []
for index, item in tqdm(enumerate(dataset['validation'])):
    train_lst.append(item)

with open("gigaword_valid.json", "w") as f:
    json.dump(train_lst, f, indent=4)

f.close()
