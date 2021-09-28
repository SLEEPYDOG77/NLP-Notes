# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/9/28 17:08
# software:PyCharm
# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/9/28 16:57
# software:PyCharm

import numpy as np
import os
import operator

class kNN(object):
    def __init__(self):
        print("This is a kNN class for Hand Writing Classify")

    def img2vector(self, filename):
        """
        将图像转换为向量
        :param filename:
        :return:
        """
        returnVect = np.zeros((1, 1024))
        fr = open(filename)
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
        return returnVect


    def classify0(self, inX, dataSet, labels, k):
        """
        k-近邻算法
        :param inX: 用于分类的输入向量
        :param dataSet: 输入的训练样本集
        :param labels: 标签向量
        :param k: 用于选择最近邻居的数目
        :return:
        """

        # 距离计算 - 欧氏距离公式
        # 获取数据集的行数
        dataSetSize = dataSet.shape[0]

        # Numpy 的 tile()函数，就是将原矩阵横向、纵向地复制
        diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5

        # 选择距离最小的k个点
        sortedDistIndicies = distances.argsort()
        classCount = {}
        for i in range(k):
            voteIlabel = labels[sortedDistIndicies[i]]
            classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

        # 排序
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

        return sortedClassCount[0][0]

    def handWritingClassTest(self):
        hwLabels = []

        # 获取目录内容
        trainingFileList = os.listdir('trainingDigits')


        m = len(trainingFileList)
        trainingMat = np.zeros((m, 1024))

        # 从文件名解析分类数字
        # 如9_45.txt - 分类是9 它是数字9的45个实例
        for i in range(m):
            fileNameStr = trainingFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            hwLabels.append(classNumStr)
            trainingMat[i, :] = self.img2vector('trainingDigits/%s' % fileNameStr)
        testFileList = os.listdir('testDigits')
        errorCount = 0.0
        mTest = len(testFileList)
        for i in range(mTest):
            fileNameStr = testFileList[i]
            fileStr = fileNameStr.split('.')[0]
            classNumStr = int(fileStr.split('_')[0])
            vectorUnderTest = self.img2vector('testDigits/%s' % fileNameStr)
            classifierResult = self.classify0(vectorUnderTest, trainingMat, hwLabels, 3)
            print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))

            if classifierResult != classNumStr:
                errorCount += 1.0
            print("\nthe total number of errors is: %d" % errorCount)
            print("\nthe total error rate is: %f" % (errorCount/float(mTest)))





if __name__ == "__main__":
    kNN = kNN()
    kNN.handWritingClassTest()
    # testVector = kNN.img2vector(filename='testDigits/0_13.txt')
    # print(testVector[0, 0:31])
    # print(testVector[0, 32:63])

