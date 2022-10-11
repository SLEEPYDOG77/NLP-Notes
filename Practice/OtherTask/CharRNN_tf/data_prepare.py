# -*- coding: utf-8 -*-
# @Time    : 2022/4/6 11:11
# @Author  : Zhang Jiaqi
# @File    : data_prepare.py
# @Description:
import json
import zhconv
from pprint import pprint

def load_data(data_dir, output_dir, mode, size):
    poetry = []
    for dir in data_dir:
        with open(dir, 'r', encoding='utf-8') as f:
            data = json.load(fp=f)
            for item in data:
                poetry.append("".join(item['paragraphs']))

    print(f"数据集诗句量为: {len(poetry)}")

    with open(output_dir + 'train.txt', 'w', encoding='utf-8') as f:
        # f.writelines(poetry)
        for item in poetry:
            item = zhconv.convert(item, 'zh-hans')
            f.write(item + '\n')

    f.close()

if __name__ == "__main__":
    data_dir = [
        '../0datasets/chinese_poetry_data/poet.song.0.json',
        '../0datasets/chinese_poetry_data/poet.song.1000.json',
        '../0datasets/chinese_poetry_data/poet.song.2000.json',
        '../0datasets/chinese_poetry_data/poet.song.3000.json',
        '../0datasets/chinese_poetry_data/poet.song.4000.json',
        '../0datasets/chinese_poetry_data/poet.song.5000.json',
        '../0datasets/chinese_poetry_data/poet.song.6000.json',
        '../0datasets/chinese_poetry_data/poet.song.7000.json',
        '../0datasets/chinese_poetry_data/poet.song.8000.json',
        '../0datasets/chinese_poetry_data/poet.song.9000.json',
        '../0datasets/chinese_poetry_data/poet.song.10000.json',
    ]
    output_dir = '../0data/chinese_poetry_data/'
    mode = 'train'
    load_data(data_dir, output_dir, mode, size=0)
