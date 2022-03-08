# -*- coding:UTF-8 -*-

# author: Zhang Jiaqi
# datetime:2021/12/24 20:52
# software:PyCharm

# 将 Monticello 句子切分成词条
sentence = """
Thomas Jefferson began building Monticello at the age of 26.
"""

print(sentence.split())
print(str.split(sentence))


# 独热向量 one-hot vector
import numpy as np
import pandas as pd

token_sequence = str.split(sentence)
# 词汇表 - 其中列举了所有想要记录的独立词条
vocab = sorted(set(token_sequence))
# print(vocab)
# print(','.join(vocab))

num_tokens = len(token_sequence)
vocab_size = len(vocab)
# 宽度 - 词汇表中独立词项的个数
# 长度 - 文档的长度
onehot_vectores = np.zeros((num_tokens, vocab_size), int)

# 对于句子中的每个词，将词汇表中与该词对应的列标记为1
for i, word in enumerate(token_sequence):
    onehot_vectores[i, vocab.index(word)] = 1

# print(' '.join(vocab))
# print(onehot_vectores)
# 每列中的数字1表示词汇表中的词出现在当前文档的当前位置
df = pd.DataFrame(onehot_vectores, columns=vocab)
# df[df == 0] = ''
print(df)
