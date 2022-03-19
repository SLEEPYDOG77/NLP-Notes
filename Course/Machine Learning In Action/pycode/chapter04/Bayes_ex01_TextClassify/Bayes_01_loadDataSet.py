# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/10/6 19:06
# software:PyCharm

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


if __name__ == "__main__":
    bayes = Bayes()
    listOPosts, listClasses = bayes.loadDataSet()
    myVocabList = bayes.createVocabList(listOPosts)
    print(myVocabList)
    print(bayes.setOfWords2Vec(myVocabList, listOPosts[0]))
    print(bayes.setOfWords2Vec(myVocabList, listOPosts[3]))
