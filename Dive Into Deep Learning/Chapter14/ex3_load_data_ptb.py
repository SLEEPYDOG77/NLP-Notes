# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 15:53
# @Author  : Zhang Jiaqi
# @File    : ex3_load_data_ptb.py
# @Description:

import math
import os
import random
from d2l.torch import DATA_HUB, DATA_URL, download_extract, Vocab, count_corpus, torch
from d2l.torch import get_dataloader_workers

# 读取数据集
DATA_HUB['ptb'] = (DATA_URL + 'ptb.zip', '319d85e578af0cdc590547f26231e4e31cdf1e42')

def read_ptb():
    data_dir = download_extract('ptb')
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

# 下采样
# 数据集中的每个词将有概率地被丢弃，该词的相对比率越高，被丢弃的概率就越大
def subsample(sentences, vocab):
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    def keep(token):
        return (random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences], counter)

# def compare_counts(token):
#     return (f'"{token}"的数量：'
#             f'之前={sum([l.count(token) for l in sentences])}，'
#             f'之后={sum([l.count(token) for l in subsampled])}')

# 中心词和上下文词的提取
def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))

            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

# 负采样
class RandomGenerator:
    def __init__(self, sampling_weights):
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

def get_negatives(all_contexts, vocab, counter, K):
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75 for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

# 小批量加载训练实例
def batchify(data):
    """返回带有负采样的跳元模型的小批量样本"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += \
            [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(
        contexts_negatives), torch.tensor(masks), torch.tensor(labels))

def load_data_ptb(batch_size, max_window_size, num_noise_words):
    num_workers = get_dataloader_workers()
    sentences = read_ptb()
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True,
        collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab

if __name__ == "__main__":
    # sentences = read_ptb()
    # vocab = Vocab(sentences, min_freq=10)
    # subsampled, counter = subsample(sentences, vocab)
    # # show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence', 'count', sentences, subsampled)
    # # print(compare_counts('the'))
    #
    # corpus = [vocab[line] for line in subsampled]
    #
    # # tiny_dataset = [list(range(7)), list(range(7, 10))]
    # # print('数据集', tiny_dataset)
    # # for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    # #     print('中心词', center, '的上下文词是', context)
    #
    # all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
    # print(f'# “中心词-上下文词对”的数量: {sum([len(contexts) for contexts in all_contexts])}')
    #
    # all_negatives = get_negatives(all_contexts, vocab, counter, 5)

    x_1 = (1, [2, 2], [3, 3, 3, 3])
    x_2 = (1, [2, 2, 2], [3, 3])
    batch = batchify((x_1, x_2))
    names = ['centers', 'contexts_negatives', 'masks', 'labels']

    data_iter, vocab = load_data_ptb(512, 5, 5)
    for batch in data_iter:
        for name, data in zip(names, batch):
            print(name, 'shape:', data.shape)
        break





