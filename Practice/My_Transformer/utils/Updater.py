# -*- coding: utf-8 -*-
# @Time    : 2022/3/17 15:14
# @Author  : Zhang Jiaqi
# @File    : Updater.py
# @Description:

import d2l.tensorflow as d2l

class Updater():  #@save
    """用小批量随机梯度下降法更新参数"""
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def __call__(self, batch_size, grads):
        d2l.sgd(self.params, grads, self.lr, batch_size)