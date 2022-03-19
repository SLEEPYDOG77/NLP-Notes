
import numpy as np
import pandas as pd

# str.split()
def tokenization_1(sentence):
    sentence_list = sentence.split()
    print(sentence_list)

# 独热向量 one-hot vectors
def tokenization_2(sentence):
    token_sequence = str.split(sentence)
    # 词条按照词库顺序进行排序 - 数字排在字母前面，大写字母排在小写字母前面
    vocab = sorted(set(token_sequence))
    print(','.join(vocab))

    num_tokens = len(token_sequence)
    vocab_size = len(vocab)

    # 宽度是词汇表中独立词项的个数
    # 长度是文档的长度
    onehot_vectors = np.zeros((num_tokens, vocab_size), int)

    # 对于句子中的每个词，将词汇表中与该词对应的列标记为1
    for index, word in enumerate(token_sequence):
        onehot_vectors[index, vocab.index(word)] = 1

    print(' '.join(vocab))
    print(onehot_vectors)

    print(pd.DataFrame(onehot_vectors, columns=vocab))

    df = pd.DataFrame(onehot_vectors, columns=vocab)
    df[df == 0] = ''
    print(df)

# 词袋向量 - python字典dict存储
def tokenization_3(sentence):
    sentence_bow = {}
    for token in sentence.split():
        sentence_bow[token] = 1
    print(sorted(sentence_bow.items()))


# 词袋向量 - pandas中的Series对象
def tokenization_4(sentence):
    df = pd.DataFrame(
        pd.Series(dict([(token, 1) for token in sentence.split()])),
        columns=['sent']
    ).T
    print(df)

# 词袋向量 - 构建词袋向量的DataFrame
def tokenization_5(sentence):
    sentence += """Construction was done mostly by local masons and carpenters.\n"""
    sentence += """He moved into the South Pvilion in 1770.\n"""
    sentence += """Turning Monticello into a neoclassica 1 masterpiece was Jefferson's obsession."""
    corpus = {}
    for index, sent in enumerate(sentence.split('\n')):
        corpus['sent{}'.format(index)] = dict((token, 1) for token in sent.split())
    df = pd.DataFrame.from_records(corpus).fillna(0).astype(int).T
    print(df[df.columns[:10]])


# 计算句子相似度 - 点积
def dot():
    v1 = pd.np.array([1, 2, 3])
    v2 = pd.np.array([2, 3, 4])
    print(v1.dot(v2))

    print((v1 * v2).sum())

# 度量词袋之间的重合度
# def

if __name__ == "__main__":
    sentence = """Thomas Jefferson began building Monticello at the age of 26."""
    # tokenization_1(sentence=sentence)

    # tokenization_2(sentence=sentence)

    # tokenization_3(sentence=sentence)

    # tokenization_4(sentence=sentence)

    tokenization_5(sentence=sentence)

    # dot()

