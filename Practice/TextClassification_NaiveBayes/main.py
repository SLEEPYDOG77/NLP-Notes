# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 15:37
# @Author  : Zhang Jiaqi
# @File    : main.py
# @Description:

import re
import os
import random
from tqdm import tqdm
from loguru import logger
from pprint import pprint
from classifier import classifier
from naivebayes import naivebayes

def getwords(doc, stopwords):
    splitter = re.compile('\W+')
    words = [s.lower() for s in splitter.split(doc) if len(s) > 2 and len(s) < 20]

    words = [word for word in words if word not in stopwords]
    return set(words)

def sampletrain(cl):
    cl.train('nobody owns the water', 'good')
    cl.train('the quick rabbit jumps fences', 'good')
    cl.train('buy phamaceuticals now', 'bad')
    cl.train('make quick money at the online casino', 'bad')
    cl.train('the quick borwn fox jumps', 'good')

def sampletrain(cl, traindata, traintarget):
    logger.info('train classifier')
    for left, right in tqdm(zip(traindata, traintarget)):
        cl.train(left, right)

def get_dataset():
    logger.info('loading dataset')
    data = []
    for root, dirs, files in os.walk(r'dataset/review_polarity/txt_sentoken/neg'):
        for file in files:
            realpath = os.path.join(root, file)
            with open(realpath, errors='ignore') as f:
                data.append((f.read(), 'bad'))
    for root, dirs, files in os.walk(r'dataset/review_polarity/txt_sentoken/pos'):
        for file in files:
            realpath = os.path.join(root, file)
            with open(realpath, errors='ignore') as f:
                data.append((f.read(), 'good'))
    random.shuffle(data)
    return data

def train_and_test_data(data_):
    logger.info('train and test data split')
    filesize = int(0.7 * len(data_))
    train_data_ = [each[0] for each in data_[:filesize]]
    train_target_ = [each[1] for each in data_[:filesize]]

    test_data_ = [each[0] for each in data_[filesize:]]
    test_target_ = [each[1] for each in data_[filesize:]]

    return train_data_, train_target_, test_data_, test_target_

if __name__ == "__main__":
    with open(r'stopwords.txt') as f:
        stopwords = f.read()

    stopwords = stopwords.split('\n')
    stopwords = set(stopwords)
    print(stopwords)

    data = get_dataset()

    cl = naivebayes(getwords, stopwords)
    train_data, train_target, test_data, test_target = train_and_test_data(data)
    sampletrain(cl, train_data, train_target)

    predict = []
    for each in test_data:
        predict.append(cl.classify(each))
    print(f'predict: {len(predict)}, test_target: {len(test_target)}')
    count = 0
    for left, right in zip(predict, test_target):
        if left == right:
            count += 1
    print(f'count: {count}')
    print(count / len(test_target))





