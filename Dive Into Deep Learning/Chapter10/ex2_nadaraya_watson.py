# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 20:37
# @Author  : Zhang Jiaqi
# @File    : ex2_nadaraya_watson.py
# @Description: Nadaraya-Watson核回归


import torch
from torch import nn
from d2l import torch as d2l
from d2l import plot, plt

n_train = 50  # 训练样本数
x_train, _ = torch.sort(torch.rand(n_train) * 5)   # 排序后的训练样本

def f(x):
    return 2 * torch.sin(x) + x**0.8

def plot_kernel_reg(y_hat):
    plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    plt.plot(x_train, y_train, 'o', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
    x_test = torch.arange(0, 5, 0.1)  # 测试样本
    y_truth = f(x_test)  # 测试样本的真实输出
    n_test = len(x_test)  # 测试样本数
    print(n_test)

    # 平均汇聚
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(y_hat)

