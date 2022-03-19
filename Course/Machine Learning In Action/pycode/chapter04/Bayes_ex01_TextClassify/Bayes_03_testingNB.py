# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/6 19:45
# software:PyCharm

import numpy as np
import math

class Bayes(object):
    def loadDataSet(self):
        """
        创建了一些实验样本
        :return:
        """
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        # 1 - 侮辱性文字
        # 0 - 正常言论
        classVec = [0, 1, 0, 1, 0, 1]
        return postingList, classVec

    def createVocabList(self, dataSet):
        """
        创建一个包含在所有文档中出现的不重复词的列表
        :param dataSet:
        :return:
        """
        # 创建一个空集
        vocabSet = set([])
        for document in dataSet:
            # 创建两个集合的并集
            vocabSet = vocabSet | set(document)

        return list(vocabSet)

    def setOfWords2Vec(self, vocabList, inputSet):
        """
        表示词汇表中的单词在输入文档中是否出现
        :param vocabList: 词汇表
        :param inputSet: 某个文档
        :return: 文档向量
        """
        # 创建一个其中所含元素都为0的向量
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] = 1
            else:
                print("the word: %s is not in my Vocabulary!" % word)
        return returnVec


    def bagOfWords2VecMN(self, vocabList, inputSet):
        """
        表示词汇表中的单词在输入文档中的出现次数
        :param vocabList: 词汇表
        :param inputSet: 某个文档
        :return: 文档向量
        """
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)] += 1
        return returnVec


    def trainNB0(self, trainMatrix, trainCategory):
        """
        朴素贝叶斯分类器训练函数
        :param trainMatrix: 文档矩阵
        :param trainCategory: 每篇文档类别标签所构成的向量
        :return:
        """
        numTrainDocs = len(trainMatrix)
        numWords = len(trainMatrix[0])
        pAbusive = sum(trainCategory) / float(numTrainDocs)

        # 初始化概率
        # p0Num = np.zeros(numWords)
        # p1Num = np.zeros(numWords)
        p0Num = np.ones(numWords)
        p1Num = np.ones(numWords)
        p0Denom = 2.0
        p1Denom = 2.0

        for i in range(numTrainDocs):
            if trainCategory[i] == 1:
                # 向量相加
                p1Num += trainMatrix[i]
                p1Denom += sum(trainMatrix[i])
            else:
                p0Num += trainMatrix[i]
                p0Denom += sum(trainMatrix[i])

        # 对每个元素做除法
        p1Vect = p1Num / p1Denom
        p0Vect = p0Num / p0Denom

        return p0Vect, p1Vect, pAbusive

    def classifyNB(self, vec2Classify, p0Vec, p1Vec, pClass1):
        p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)
        p0 = sum(vec2Classify * p0Vec) + math.log(1.0 - pClass1)
        if p1 > p0:
            return 1
        else:
            return 0

    def testingNB(self):
        listOPosts, listClasses = self.loadDataSet()
        myVocabList = self.createVocabList(listOPosts)
        trainMat = []
        for postinDoc in listOPosts:
            trainMat.append(self.setOfWords2Vec(myVocabList, postinDoc))

        p0V, p1V, pAb = self.trainNB0(np.array(trainMat), np.array(listClasses))
        testEntry = ['love', 'my', 'dalmation']
        thisDoc = np.array(self.setOfWords2Vec(myVocabList, testEntry))
        print(testEntry, 'classified as: ', self.classifyNB(thisDoc, p0V, p1V, pAb))
        testEntry = ['stupid', 'garbage']
        thisDoc = np.array(self.setOfWords2Vec(myVocabList, testEntry))
        print(testEntry, 'classified as: ', self.classifyNB(thisDoc, p0V, p1V, pAb))



if __name__ == "__main__":
    bayes = Bayes()
    bayes.testingNB()
    # listOPosts, listClasses = bayes.loadDataSet()
    # myVocabList = bayes.createVocabList(listOPosts)
    # trainMat = []
    # for postinDoc in listOPosts:
    #     trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
    # p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
    # print(p0V)
    # print(p1V)
    # print(pAb)
