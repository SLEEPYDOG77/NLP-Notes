# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/7 21:01
# software:PyCharm

import numpy as np

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


if __name__ == "__main__":
    logRegres = LogRegres()
    dataArr, labelMat = logRegres.loadDataSet()
    print(logRegres.gradAscent(dataArr, labelMat))
