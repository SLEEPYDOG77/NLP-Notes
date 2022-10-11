# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 15:35
# @Author  : Zhang Jiaqi
# @File    : classifier.py
# @Description:

class classifier():
    def __init__(self, getfeatures, stopwords):
        # 记录位于各分类中不同特征的数量
        # {'python': {'bad': 0, 'good': 6}, 'money': {'bad': 5, 'good': 1}}
        self.fc = {}
        # 记录各分类被使用次数的词典
        self.cc = {}
        # 函数 - 从即将被归类的文档中提取出特征
        self.getfeatures = getfeatures
        # 停用词表
        self.stopwords = stopwords
        print('init')

    # 增加对特征 / 分类组合的计数值
    def incf(self, f, cat):
        self.fc.setdefault(f, {})
        self.fc[f].setdefault(cat, 1)
        self.fc[f][cat] += 1

    # 增加某一个分类的计数值
    def incc(self, cat):
        self.cc.setdefault(cat, 1)
        self.cc[cat] += 1

    # 计算某一个特征在某一个分类中出现的次数
    def fcount(self, f, cat):
        if f in self.fc and cat in self.fc[f]:
            return self.fc[f][cat]
        else:
            return 0.0

    # 属于某一个分类的文档总数
    def catcount(self, cat):
        if cat in self.cc:
            return self.cc[cat]
        return 0

    # 所有的文档总数
    def totalcount(self):
        return sum(self.cc.values())

    # 所有文档的种类
    def categories(self):
        return self.cc.keys()

    def train(self, item, cat):
        features = self.getfeatures(item, self.stopwords)
        for f in features:
            self.incf(f, cat)
        self.incc(cat)

    def fprob(self, f, cat):
        if self.catcount(cat) == 0:
            return 0

        # 特征在该分类中出现的次数 / 该特征下文档的总数目
        return self.fcount(f, cat) / self.catcount(cat)

    def weightedprob(self, f, cat, prf, weight=1, ap=0.5):
        # 使用fprob函数计算原始的条件概率
        basicprob = prf(f, cat)
        totals = sum([self.fcount(f, c) for c in self.categories()])
        bp = ((weight * ap) + (totals * basicprob)) / (weight + totals)
        return bp
