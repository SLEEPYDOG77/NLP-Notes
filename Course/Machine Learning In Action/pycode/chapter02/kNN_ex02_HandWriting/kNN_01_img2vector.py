# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/9/28 16:57
# software:PyCharm

import numpy as np

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


if __name__ == "__main__":
    kNN = kNN()
    testVector = kNN.img2vector(filename='testDigits/0_13.txt')
    print(testVector[0, 0:31])
    print(testVector[0, 32:63])

