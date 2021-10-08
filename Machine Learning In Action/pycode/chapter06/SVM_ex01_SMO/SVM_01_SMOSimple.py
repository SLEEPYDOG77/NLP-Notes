# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/8 10:25
# software:PyCharm
import random
import numpy as np

class SMO(object):
    def loadDataSet(self, fileName):
        """
        打开文件并逐行解析，从而得到每行的类标签和整个数据矩阵
        :param fileName:
        :return:
        """
        dataMat = []
        labelMat = []
        fr = open(fileName)

        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
        return dataMat, labelMat

    def selectJrand(self, i, m):
        """

        :param i: 第一个alpha的下标
        :param m: 所有alpha的数目
        :return:
        """
        j = i
        while j == i:
            j = int(random.uniform(0, m))
        return j

    def clipAlpha(self, aj, H, L):
        """
        用于调整大于H或小于L的alpha值
        :param aj:
        :param H:
        :param L:
        :return:
        """
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def smoSimple(self, dataMatIn, classLabels, C, toler, maxIter):
        """
        简化版SMO算法
        :param dataMatIn: 数据集
        :param classLabels: 类别标签
        :param C: 常数C
        :param toler: 容错率
        :param maxIter: 退出前最大的循环次数
        :return:
        """
        dataMatrix = np.mat(dataMatIn)
        labelMat = np.mat(classLabels).transpose()

        b = 0
        m, n = np.shape(dataMatrix)
        alphas = np.mat(np.zeros((m, 1)))

        iter = 0

        while iter < maxIter:
            alphaPairsChaged = 0

            for i in range(m):
                fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
                Ei = fXi - float(labelMat[i])

                # 如果alpha可以更改进入优化过程
                if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):

                    # 随机选择第二个alpha
                    j = self.selectJrand(i, m)
                    fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                    Ej = fXj - float(labelMat[j])
                    alphaIold = alphas[i].copy()
                    alphaJold = alphas[j].copy()

                    # 保证alpha在0和C之间
                    if (labelMat[i] != labelMat[j]):
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])

                    if L == H:
                        print("L==H")
                        continue
                    eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[j, :] * dataMatrix[j, :].T

                    if eta >= 0:
                        print("eta>=0")
                        continue

                    alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                    alphas[j] = self.clipAlpha(alphas[j], H, L)
                    if abs(alphas[j] - alphaJold) < 0.0001:
                        print("j not moving enough")
                        continue

                    # 对i进行修改，修改量与j相同，但方向相反
                    alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])

                    b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * \
                         dataMatrix[i, :] * dataMatrix[i, :].T - \
                         labelMat[j] * (alphas[j] - alphaJold) *\
                         dataMatrix[i, :] * dataMatrix[j, :].T

                    # 设置常数项
                    b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * \
                         dataMatrix[i, :] * dataMatrix[j, :].T - \
                         labelMat[j] * (alphas[j] - alphaJold) * \
                         dataMatrix[j, :] * dataMatrix[j, :].T

                    if (0 < alphas[i]) and (C > alphas[i]):
                        b = b1
                    elif (0 < alphas[j]) and (C > alphas[j]):
                        b = b2
                    else: b = (b1 + b2) / 2.0
                    alphaPairsChaged += 1
                    print("iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChaged))

            if (alphaPairsChaged == 0):
                iter += 1
            else:
                iter = 0
            print("iteration number: %d" % iter)
        return b, alphas





if __name__ == "__main__":
    svm = SMO()
    dataArr, labelArr = svm.loadDataSet('testSet.txt')
    # print(labelArr)
    b, alphas = svm.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
    print(b)
    print(alphas)
    print(np.shape(alphas[alphas > 0]))
    for i in range(100):
        if alphas[i] > 0.0:
            print(dataArr[i], labelArr[i])


