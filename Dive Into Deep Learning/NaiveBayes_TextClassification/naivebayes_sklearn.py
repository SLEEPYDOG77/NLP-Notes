# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 20:47
# @Author  : Zhang Jiaqi
# @File    : naivebayes_sklearn.py
# @Description: sklearn自带的贝叶斯分类器进行文本分类和参数调优

import os
import random
from loguru import logger
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB

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


def multinomialNB_train(train_data, train_target, test_data, test_target):
    nbc = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', MultinomialNB(alpha=1.0)),
    ])

    nbc.fit(train_data, train_target)
    predict = nbc.predict(test_data)
    count = 0
    for left, right in zip(predict, test_target):
        if left == right:
            count += 1
    print(f'多项式模型贝叶斯分类器：{count / len(test_target)}')


def BernoulliNB_train(train_data, train_target, test_data, test_target):
    nbc_1 = Pipeline([
        ('vect', TfidfVectorizer()),
        ('clf', BernoulliNB(alpha=0.1)),
    ])
    nbc_1.fit(train_data, train_target)
    predict = nbc_1.predict(test_data)  # 在测试集上预测结果
    count = 0  # 统计预测正确的结果个数
    for left, right in zip(predict, test_target):
        if left == right:
            count += 1
    print(f'伯努利模型分类器：{count / len(test_target)}')


if __name__ == "__main__":
    data = get_dataset()
    train_data, train_target, test_data, test_target = train_and_test_data(data)
    multinomialNB_train(train_data, train_target, test_data, test_target)
    BernoulliNB_train(train_data, train_target, test_data, test_target)
