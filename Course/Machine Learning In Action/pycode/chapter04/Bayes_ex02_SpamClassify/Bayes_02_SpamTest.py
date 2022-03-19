# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/7 20:29
# software:PyCharm
import random
import re
import math
import numpy as np

class SpamBayes(object):
    def __init__(self):
        pass

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

    def textParse(self, bigString):
        """
        接受一个大写字符串并将其解析为字符串列表
        :param bigString:
        :return:
        """
        listOfTokens = re.split(r'\W+', bigString)
        # 去掉少于两个字符的字符串，并将所有字符串转换为小写
        return [tok.lower() for tok in listOfTokens if len(tok) > 2]


    def spamTest(self):
        """
        对贝叶斯垃圾邮件分类器进行自动化处理
        :return: 输出在10封随机选择的电子邮件上的分类错误率
        """
        docList = []
        classList = []
        fullText = []

        # 导入并解析文本
        for i in range(1, 26):
            wordList = self.textParse(open('email/spam/%d.txt' % i).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)

            wordList = self.textParse(open('email/ham/%d.txt' % i).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)

        # 随机构建训练集 - 留存交叉验证
        vocabList = self.createVocabList(docList)
        trainingSet = list(range(50))
        testSet = []
        for i in range(10):
            randIndex = int(random.uniform(0, len(trainingSet)))
            testSet.append(trainingSet[randIndex])
            del(trainingSet[randIndex])

        trainMat = []
        trainClasses = []

        for docIndex in trainingSet:
            trainMat.append(self.setOfWords2Vec(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])

        p0V, p1V, pSpam = self.trainNB0(np.array(trainMat), np.array(trainClasses))
        errorCount = 0

        # 对测试集分类
        for docIndex in testSet:
            wordVector = self.setOfWords2Vec(vocabList, docList[docIndex])
            if self.classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
                errorCount += 1

        print('the error rate is: ', float(errorCount) / len(testSet))

if __name__ == "__main__":
    bayes = SpamBayes()
    bayes.spamTest()
