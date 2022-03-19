# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/9/30 16:12
# software:PyCharm

import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制带箭头的注解
    :param nodeTxt:
    :param centerPt:
    :param parentPt:
    :param nodeType:
    :return:
    """
    createPlot.ax1.annotate(nodeTxt,
                            xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    """
    使用文本注解绘制树节点
    :return:
    """
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('决策节点', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('叶节点', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

