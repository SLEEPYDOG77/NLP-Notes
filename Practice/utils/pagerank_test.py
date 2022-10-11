# -*- coding: utf-8 -*-
# @Time    : 2022/4/13 9:45
# @Author  : Zhang Jiaqi
# @File    : pagerank_test.py
# @Description: 鼎鼎大名的PageRank算法——理论+实战 - 王鹏的文章 - 知乎
# https://zhuanlan.zhihu.com/p/113570465


import numpy as np
import networkx as nx

def nodes2matrix(node_json):
    node2id = {}
    dim = len(node_json)
    for index, key in enumerate(node_json.keys()):
        node2id[key] = index

    matrix = np.zeros((dim, dim))

    for key in node_json.keys():
        nodeid = node2id[key]

        for neighbor in node_json[key]:
            neighbor = node2id[neighbor]
            matrix[neighbor][nodeid] = 1

    for i in range(dim):
        matrix[:, i] = matrix[:, i] / sum(matrix[:, i])

    return matrix


def pagerank(matrix, iter=10, d=0.85):
    """
    设置各节点PR的初始值，然后对公式不断迭代，直到收敛
    :param matrix: 有向图构建转移矩阵M
    :param iter: 迭代次数
    :param d: 阻尼系数
    :return:
    """
    length = len(matrix[0])
    inital_value = np.ones(length) / length
    pagerank_value = inital_value

    for i in range(iter):
        pagerank_value = matrix @ pagerank_value * d + (1 - d) / length
        print("iter {}: the pr value is {}".format(i, pagerank_value))

    return pagerank_value


def pagerank_ntx():
    G = nx.DiGraph()

    pass


if __name__ == "__main__":
    node_json = {
        "A": [
            "B",
            "C"
        ],
        "B": [
            "A",
            "C"
        ],
        "C": [
            "D",
            "B"
        ],
        "D": [
            "A",
            "B"
        ]
    }

    matrix = nodes2matrix(node_json)
    print(matrix)
    print("the matrix is: ", matrix)

    pagerank(matrix)
