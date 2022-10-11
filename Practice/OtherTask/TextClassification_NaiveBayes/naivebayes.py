# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 19:19
# @Author  : Zhang Jiaqi
# @File    : naivebayes.py
# @Description:

from classifier import classifier

class naivebayes(classifier):
    def __init__(self, getfeatures, stopwords):
        classifier.__init__(self, getfeatures, stopwords)

    def docprob(self, item, cat):
        features = self.getfeatures(item, self.stopwords)

        p = 1
        for f in features:
            p *= self.weightedprob(f, cat, self.fprob)
        return p

    def prob(self, item, cat):
        """
        计算一篇文档 属于某一分类的概率
        :param item: 文档
        :param cat: 某一分类
        :return: 概率
        """
        # P(category | Document) = P(Document | category) * P(category) / P(Document)
        # P(Document) - 对于所有文档来说都一样 可以忽略
        # P(Document | category) * P(category) ->
        #     P(Document | category) -> docprob(item, cat)
        #     P(category) -> (catcount(cat) / totalcount())
        docprob = self.docprob(item, cat)
        catprob = self.catcount(cat) / self.totalcount()
        return docprob * catprob

    def classify(self, item):
        max = 0.0
        best = list(self.categories())[0]
        probs = {}
        for cat in self.categories():
            probs[cat] = self.prob(item, cat)
            print(probs[cat])
            if probs[cat] > max:
                max = probs[cat]
                best = cat
        return best