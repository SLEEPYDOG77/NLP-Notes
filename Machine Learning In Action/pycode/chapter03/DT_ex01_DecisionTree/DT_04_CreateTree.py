# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/9/30 16:00
# software:PyCharm

from math import log
import operator

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
        # 函数中调用的数据需要满足一定的要求:
        # 1. 数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度
        # 2. 数据的最后一列或者每个实例的最后一个元素是当前实例的列表标签
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


    def majorityCnt(self, classList):
        """
        返回出现此处最多的分类名称
        :param classList:
        :return:
        """
        classCount = {}
        for vote in classList:
            if vote not in classCount.keys():
                classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def createTree(self, dataSet, labels):
        """
        创建决策树
        :param dataSet: 数据集
        :param labels: 标签列表
        :return:
        """
        # 递归函数的两个停止条件：
        # 1. 所有的类标签完全相同
        # 2. 使用完了所有特征，仍然不能将数据集划分成仅包含唯一类别的分组2
        classList = [example[-1] for example in dataSet]
        # 类别完全相同则停止继续划分
        if classList.count(classList[0]) == len(classList):
            return classList[0]

        # 遍历完所有特征时返回出现次数最多的类别
        if len(dataSet[0]) == 1:
            return self.majorityCnt(classList=classList)

        bestFeat = self.chooseBestFeatureToSplit(dataSet=dataSet)
        bestFeatLabel = labels[bestFeat]

        # 字典类型存储树
        myTree = {bestFeatLabel: {}}

        del(labels[bestFeat])

        # 得到列表包含的所有属性值
        featValues = [example[bestFeat] for example in dataSet]
        uniqueVals = set(featValues)
        for value in uniqueVals:
            subLabels = labels[:]
            myTree[bestFeatLabel][value] = self.createTree(self.splitDataSet(dataSet, bestFeat, value), subLabels)

        return myTree



if __name__ == "__main__":
    trees = Trees()
    myDat, labels = trees.createDataSet()

    myTree = trees.createTree(myDat, labels)
    print(myTree)



