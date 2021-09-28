from numpy import *
import operator

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

        # Numpy 的 tile()函数，就是将原矩阵横向、纵向地复制
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



if __name__ == "__main__":
    kNN = kNN()
    group, labels = kNN.createDataSet()

    sortedClassCount = kNN.classify0([0, 0], group, labels, 3)
    print(sortedClassCount)