# -*- coding: utf-8 -*-
# @Time    : 2022/3/4 20:25
# @Author  : Zhang Jiaqi
# @File    : ex4_word2vec.py
# @Description: 14.4 预训练word2vec

import math
import torch
from torch import nn
from d2l.torch import load_data_ptb

# 定义前向传播
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred


# 二元交叉熵损失
class SigmoidBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none"
        )
        return out.mean(dim=1)




if __name__ == "__main__":
    batch_size, max_window_size, num_noise_words = 512, 5, 5
    data_iter, vocab = load_data_ptb(batch_size, max_window_size, num_noise_words)

    embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
    print(f'Parameter embedding_weight({embed.weight.shape},'
          f'dtype={embed.weight.dtype})')

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(embed(x))
