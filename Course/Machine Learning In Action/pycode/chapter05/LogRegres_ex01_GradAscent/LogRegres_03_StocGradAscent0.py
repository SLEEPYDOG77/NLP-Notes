# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/7 21:25
# software:PyCharm

import numpy as np
import matplotlib.pyplot as plt

class LogRegres(object):
    def __init__(self):
        pass

    def loadDataSet(self):
        dataMat = []
        labelMat = []
        fr = open('testSet.txt')
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            labelMat.append(int(lineArr[2]))

        return dataMat, labelMat

    def sigmoid(self, inX):
        return 1.0 / (1 + np.exp(-inX))

    def gradAscent(self, dataMatIn, classLabels):
        """
        梯度上升算法
        :param dataMatIn: 每列分别代表每个不同的特征，每行代表每个样本
        :param classLabels: 类别标签
        :return:
        """
        dataMattrix = np.mat(dataMatIn)
        labelMat = np.mat(classLabels).transpose()

        m, n = np.shape(dataMattrix)

        # 向目标移动的步长
        alpha = 0.001
        # 迭代次数
        maxCycles = 500
        weights = np.ones((n, 1))

        for k in range(maxCycles):
            h = self.sigmoid(dataMattrix * weights)
            error = (labelMat - h)
            weights = weights + alpha * dataMattrix.transpose() * error

        return weights

    def plotBestFit(self, weights):
        """
        用matplotlib画出数据集和logistic回归最佳拟合直线的函数
        :param weights:
        :return:
        """
        dataMat, labelMat = self.loadDataSet()
        dataArr = np.array(dataMat)

        n = np.shape(dataArr)[0]

        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []

        for i in range(n):
            if int(labelMat[i]) == 1:
                xcord1.append(dataArr[i, 1])
                ycord1.append(dataArr[i, 2])
            else:
                xcord2.append(dataArr[i, 1])
                ycord2.append(dataArr[i, 2])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')

        x = np.arange(-3.0, 3.0, 0.1)
        y = (-weights[0] - weights[1]*x) / weights[2]
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()


    def stocGradAscent0(self, dataMatrix, classLabels):
        """
        随机梯度上升算法
        :param dataMatrix:
        :param classLabels:
        :return:
        """
        m, n = np.shape(dataMatrix)
        alpha = 0.01
        weights = np.ones(n)
        for i in range(m):
            h = self.sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i]
        return weights


if __name__ == "__main__":
    logRegres = LogRegres()
    dataArr, labelMat = logRegres.loadDataSet()
    # print(logRegres.gradAscent(dataArr, labelMat))
    weights = logRegres.stocGradAscent0(np.array(dataArr), labelMat)
    logRegres.plotBestFit(weights)
