# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/8 9:52
# software:PyCharm

import numpy as np
import random

class HorseColic(object):
    def __init__(self):
        pass

    def sigmoid(self, inX):
        return 1.0 / (1 + np.exp(-inX))

    def stocGradAscent1(self, dataMatrix, classLabels, numIter = 150):
        """
        改进的随机梯度上升算法
        :param dataMatrix:
        :param classLabels:
        :param numIter:
        :return:
        """
        m, n = np.shape(dataMatrix)
        weights = np.ones(n)

        for j in range(numIter):
            dataIndex = list(range(m))
            for i in range(m):
                # alpha 每次迭代时都需要调整
                alpha = 4 / (1.0 + j + i) + 0.01
                # 随机选取更新
                randIndex = int(random.uniform(0, len(dataIndex)))
                h = self.sigmoid(sum(dataMatrix[randIndex]*weights))
                error = classLabels[randIndex] - h
                weights = weights + alpha * error * dataMatrix[randIndex]
                del(dataIndex[randIndex])

        return weights

    def classifyVector(self, inX, weights):
        """
        计算对应的sigmoid值
        :param inX: 回归系数
        :param weights: 特征向量
        :return:
        """
        prob = self.sigmoid(sum(inX * weights))
        if prob > 0.5:
            return 1.0
        else:
            return 0.0


    def colicTest(self):
        """
        打开测试集和训练集，并对数据进行格式化处理
        :return:
        """
        frTrain = open('horseColicTraining.txt')
        frTest = open('horseColicTest.txt')

        trainingSet = []
        trainingLabels = []

        for line in frTrain.readlines():
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            trainingSet.append(lineArr)
            trainingLabels.append(float(currLine[21]))
        trainWeights = self.stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
        errorCount = 0
        numTestVec = 0.0

        for line in frTest.readlines():
            numTestVec += 1.0
            currLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(currLine[i]))
            if int(self.classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
                errorCount += 1

        errorRate = (float(errorCount) / numTestVec)

        print("the error rate of this test is: %f" % errorRate)
        return errorRate

    def multiTest(self):
        """
        调用函数colicTest() 10次并求结果的平均值
        :return:
        """
        numTests = 10
        errorSum = 0.0
        for k in range(numTests):
            errorSum += self.colicTest()
        print("after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests)))


if __name__ == "__main__":
    horseColic = HorseColic()
    horseColic.multiTest()
