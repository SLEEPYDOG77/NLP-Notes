from numpy import *
import operator

import matplotlib
import matplotlib.pyplot as plt

class kNN(object):
    def __init__(self):
        print("This is a kNN class.")

    def createDataSet(self):
        """
        创建数据集
        :return:
        """
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        return group, labels


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
        diffMat = tile(inX, (dataSetSize, 1)) - dataSet
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


    def file2matrix(self, filename):
        """
        从文本文件中解析数据
        :param filename: 文件名
        :return: numPy矩阵
        """
        # 打开文件 得到文件的行数
        fr = open(filename)
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)

        print("numberOfLines", numberOfLines)

        # 创建要用作返回的 numPy矩阵
        returnMat = zeros((numberOfLines, 3))

        classLabelVector = []
        index = 0

        # 解析文件数据到列表
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index, 0:3] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat, classLabelVector


    def matplotlib(self, datingDataMat, datingLabels):
        """
        使用Matplotlib制作原始数据的散点图
        :param datingDataMat:
        :param datingLabels:
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*array(datingLabels), 15.0*array(datingLabels))
        plt.show()


if __name__ == "__main__":
    kNN = kNN()
    datingDataMat, datingLabels = kNN.file2matrix(filename='datingTestSet2.txt')
    # print("datingDataMat")
    # print(datingDataMat)
    #
    # print("datingLabels[0:20]")
    # print(datingLabels[0:20])

    kNN.matplotlib(datingDataMat=datingDataMat, datingLabels=datingLabels)


