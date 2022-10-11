# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 19:52
# @Author  : Zhang Jiaqi
# @File    : data_prepare.py
# @Description:

import json
from lxml import etree
from tqdm import tqdm

# we follow the previous research to split the dataset for traning, validation and test,
# 2.4M sentence pairs for training
# 8K for validation
# 0.7K for testing

def data_preprocess_lcsts(data_dir, output_dir, mode, size):
    summary_array = []
    short_text_array = []

    if mode == 'test' or mode == 'valid':
        with open(data_dir, "r", encoding='UTF-8') as f:
            data = f.read()
            source = etree.HTML(data)
            doc_s = source.xpath('//doc')
            print(f"Length of source 0data: {len(doc_s)}")
            for index, doc in enumerate(doc_s):
                if index < size:
                    try:
                        summary = '<s>' + doc.xpath('./summary/text()')[0].replace(' ', '').replace('\n', '').replace('UNK', '<unk>') + '</s>'
                        short_text = '<s>' + doc.xpath('./short_text/text()')[0].replace(' ', '').replace('\n', '').replace('UNK', '<unk>') + '</s>'
                        summary_array.append(summary)
                        short_text_array.append(short_text)
                    except Exception as err:
                        print(err)

    elif mode == 'train':
        with open(data_dir, "r", encoding="UTF-8") as f:
            source = json.load(f)
            print(f"Length of source 0data: {len(source)}")
            for index, item in enumerate(source):
                if index < size:
                    summary_array.append(item['title'])
                    short_text_array.append(item['content'])

    print(f"Length of summary_array: {len(summary_array)}")
    print(f"Length of short_text_array: {len(short_text_array)}")

    for index in tqdm(range(len(summary_array))):
        with open(f"{output_dir}/{mode}.tgt", "a", encoding='UTF-8') as f:
            f.write(str(summary_array[index]))
            f.write('\n')

        with open(f"{output_dir}/{mode}.src", "a", encoding='UTF-8') as f:
            f.write(str(short_text_array[index]))
            f.write('\n')


def data_preprocess_giga(data_dir, output_dir, mode, size):
    summary_array = []
    short_text_array = []

    with open(data_dir, "r", encoding="UTF-8") as f:
        source = json.load(f)
        print(f"Length of source 0data: {len(source)}")
        for index, item in enumerate(source):
            if index < size:
                summary = '<s> ' + item['summary'].replace('\n', '').replace('UNK', '<unk>') + ' </s>'
                short_text = '<s> ' + item['document'].replace('\n', '').replace('UNK', '<unk>') + ' </s>'
                summary_array.append(summary)
                short_text_array.append(short_text)

    for index in tqdm(range(len(summary_array))):
        with open(f"{output_dir}/{mode}.tgt", "a", encoding='UTF-8') as f:
            f.write(str(summary_array[index]))
            f.write('\n')

        with open(f"{output_dir}/{mode}.src", "a", encoding='UTF-8') as f:
            f.write(str(short_text_array[index]))
            f.write('\n')


if __name__ == "__main__":
    # # train
    # train_data_dir = '../0datasets/lcsts_data/lcsts_data.json'
    # output_dir = '../0data/lcsts_data/data_prepared_test'
    # mode = 'train'
    # train_size = 20000
    # data_preprocess(train_data_dir, output_dir, mode, train_size)
    #
    # # valid
    # valid_data_dir = '../0datasets/lcsts_data/' + 'PART_II.txt'
    # mode = 'valid'
    # valid_size = 10000
    # data_preprocess(valid_data_dir, output_dir, mode, valid_size)
    #
    # # test
    # test_data_dir = '../0datasets/lcsts_data/' + 'PART_III.txt'
    # mode = 'test'
    # test_size = 1000
    # data_preprocess(test_data_dir, output_dir, mode, test_size)

    # train
    train_data_dir = '../0datasets/gigaword/gigaword_train.json'
    output_dir = '../0data/gigaword_data/data_prepared_test'
    mode = 'train'
    train_size = 10000
    data_preprocess_giga(train_data_dir, output_dir, mode, train_size)

    # valid
    train_data_dir = '../0datasets/gigaword/gigaword_valid.json'
    mode = 'valid'
    train_size = 10000
    data_preprocess_giga(train_data_dir, output_dir, mode, train_size)

    # test
    train_data_dir = '../0datasets/gigaword/gigaword_test.json'
    mode = 'test'
    train_size = 1000
    data_preprocess_giga(train_data_dir, output_dir, mode, train_size)
