# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/10 9:59
# software:PyCharm

import numpy as np
import random

def kernelTrans(X, A, kTup):
    """
    核转换函数
    :param X:
    :param A:
    :param kTup:
    :return:
    """
    m, n = np.shape(X)
    K = np.mat(np.zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1 * kTup[1]**2))
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not Recognized')
    return K

class optStruct(object):
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # 误差缓存
        # 第一列是eCache是否有效的标志位
        # 第二列给出的是实际的E值
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.n)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)


class SVM(object):
    def __init__(self):
        pass

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


    def calcEk(self, oS, k):
        fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
        Ek = fXk - float(oS.labelMat[k])
        return Ek


    def selectJ(self, i, oS, Ei):
        """
        用于选择第二个alpha或者说内循环的alpha值2
        :param i:
        :param oS:
        :param Ei:
        :return:
        """
        maxK = -1
        maxDeltaE = 0
        Ej = 0

        oS.eCache[i] = [1, Ei]
        validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
        if (len(validEcacheList)) > 1:
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(oS, k)
                deltaE = abs(Ei - Ek)
                # 选择具有最大步长的j
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            j = self.selectJrand(i, oS.m)
            Ej = self.calcEk(oS, j)

        return j, Ej

    def updateEk(self, oS, k):
        """
        计算误差值并存入缓存当中
        :param oS:
        :param k:
        :return:
        """
        Ek = self.calcEk(oS, k)
        oS.eCache[k] = [1, Ek]


    def innerL(self, i, oS):
        Ei = self.calcEk(oS, i)
        if ((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
                ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0)):
            j, Ej = self.selectJ(i, oS, Ei)
            alphaIold = oS.alphas[i].copy()
            alphaJold = oS.alphas[j].copy()

            if (oS.labelMat[i] != oS.labelMat[j]):
                L = max(0, oS.alphas[j] - oS.alphas[i])
                H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
            else:
                L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
                H = min(oS.C, oS.alphas[j] + oS.alphas[i])
            if L == H:
                print("L==H")
                return 0

            # eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
            eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]

            if eta >= 0:
                print("eta >= 0")
                return 0

            oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
            oS.alphas[j] = self.clipAlpha(oS.alphas[j], H, L)
            self.updateEk(oS, j)

            if (abs(oS.alphas[j] - alphaJold) < 0.00001):
                print("j not moving enough")
                return 0

            oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
            self.updateEk(oS, i)

            # b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
            #     oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * \
            #     (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
            #
            # b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * \
            #     oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * \
            #     (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T

            b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
                oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]

            b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
                oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]

            if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
                oS.b = b1
            elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0


    def smoP(self, dataMatIn, classLabels, C, toler, maxIter, kTup=('lin', 0)):
        """
        寻找决策边界的优化例程
        :param self:
        :param dataMatIn: 数据集
        :param classLabels: 类别标签
        :param C: 常数C
        :param toler: 容错率
        :param maxIter: 退出前最大的循环次数
        :param kTup:
        :return:
        """
        oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)
        iter = 0
        entireSet = True
        alphaPairsChanged = 0

        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:
                for i in range(oS.m):
                    alphaPairsChanged += self.innerL(i, oS)
                    print("fullSet, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1

            else:
                nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i, oS)
                    print("non-bound, iter: %d i: %d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1

            if entireSet:
                entireSet = False
            elif (alphaPairsChanged == 0):
                entireSet = True
            print("iteration number: %d" % iter)

        return oS.b, oS.alphas

    def calcWs(self, alphas, dataArr, classLabels):
        X = np.mat(dataArr)
        labelMat = np.mat(classLabels).transpose()
        m, n = np.shape(X)
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
        return w

    def testRbf(self, k1=1.3):
        dataArr, labelArr = self.loadDataSet('testSetRBF.txt')
        b, alphas = self.smoP(dataArr, labelArr, 200, 0.0001, 1000, ('rbf', k1))
        dataMat = np.mat(dataArr)
        labelMat = np.mat(labelArr).transpose()
        svInd = np.nonzero(alphas.A > 0)[0]

        # 构建支持向量矩阵
        sVs = dataMat[svInd]
        labelSV = labelMat[svInd]

        print("there are %d Support Vectors" % np.shape(sVs)[0])
        m, n = np.shape(dataMat)

        errorCount = 0
        for i in range(m):
            # 利用kernelTrans函数得到转换后的数据
            kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
            # 再用其与前面的alpha及类别标签值求积
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
            if np.sign(predict) != np.sign(labelArr[i]):
                errorCount += 1

        print("the training error rate is: %f" % (float(errorCount) / m))


        dataArr, labelArr = self.loadDataSet('testSetRBF2.txt')
        errorCount = 0
        dataMat = np.mat(dataArr)
        labelMat = np.mat(labelArr).transpose()
        m, n = np.shape(dataMat)

        for i in range(m):
            kernelEval = kernelTrans(sVs, dataMat[i, :], ('rbf', k1))
            predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
            if np.sign(predict) != np.sign(labelArr[i]):
                errorCount += 1
        print("the test error rate is: %f" % (float(errorCount) / m))





if __name__ == "__main__":
    svm = SVM()
    # dataArr, labelArr = svm.loadDataSet('testSet.txt')
    # b, alphas = svm.smoP(dataArr, labelArr, 0.6, 0.001, 40)



