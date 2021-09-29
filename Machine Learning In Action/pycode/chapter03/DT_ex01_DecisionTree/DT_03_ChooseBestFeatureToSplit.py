# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/9/29 19:28
# software:PyCharm
from math import log

class Trees(object):
    def __init__(self):
        print("This is a Decision Tree Test.")

    def calcShannonEnt(self, dataSet):
        """
        计算给定数据集的香农熵
        :param dataSet:
        :return:
        """
        numEntries = len(dataSet)
        labelCounts = {}

        # 为所有可能分类创建字典
        for featVec in dataSet:
            currentLabel = featVec[-1]
            if currentLabel not in labelCounts.keys():
                labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0

        # 以2为底求对数
        for key in labelCounts:
            prob = float(labelCounts[key])/numEntries
            shannonEnt -= prob * log(prob, 2)

        return shannonEnt


    def createDataSet(self):
        """
        创建数据集
        :return:
        """
        dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
        labels = ['no surfacing', 'flippers']
        return dataSet, labels


    def splitDataSet(self, dataSet, axis, value):
        """
        按照给定特征划分数据集
        :param dataSet: 待划分的数据集
        :param axis: 划分数据集的特征
        :param value: 需要返回的特征的值
        :return:
        """

        # 创建新的list对象
        retDataSet = []
        for featVec in dataSet:
            # 抽取
            if featVec[axis] == value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    def chooseBestFeatureToSplit(self, dataSet):
        """
        实现选取特征 划分数据集 计算得出最好的划分数据集的特诊
        :param dataSet:
        :return:
        """
        numFeatures = len(dataSet[0]) - 1
        baseEntropy = self.calcShannonEnt(dataSet)
        bestInfoGain = 0.0
        bestFeature = -1

        for i in range(numFeatures):
            # 创建唯一的分类标签列表
            featList = [example[i] for example in dataSet]
            uniqueVals = set(featList)
            newEntropy = 0.0

            for value in uniqueVals:
                # 计算每种划分方式的信息熵
                subDataSet = self.splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                # 计算最好的信息增益
                bestInfoGain = infoGain
                bestFeature = i
        return bestFeature


if __name__ == "__main__":
    trees = Trees()
    myDat, labels = trees.createDataSet()
    # print(trees.splitDataSet(myDat, 0, 1))
    # print(trees.splitDataSet(myDat, 0, 0))
    # print("myDat")
    # print(myDat)
    #
    # print("labels")
    # print(labels)
    #
    # result = trees.calcShannonEnt(myDat)
    # print("result", result)



