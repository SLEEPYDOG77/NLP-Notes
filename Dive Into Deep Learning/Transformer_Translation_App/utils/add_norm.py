# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 19:02
# @Author  : Zhang Jiaqi
# @File    : add_norm.py
# @Description:

from torch import nn

# 残差连接和层规范化
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
