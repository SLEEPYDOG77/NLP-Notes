# -*- coding: utf-8 -*-
# @Time    : 2022/5/7 21:15
# @Author  : Zhang Jiaqi
# @File    : rnn_text_generation.py
# @Description: RNN模型与NLP应用(69)：Text Generation (自动文本生成) - 知乎
# https://www.zhihu.com/zvideo/1399369522768314368

import torch.nn
import numpy as np

def load_lcsts():
    filepath = '../0data/lcsts_data/data_prepared_test/train.src'
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(len(lines))
    text = ''.join([line.replace('\n', '') for line in lines])
    print(len(text))
    print(text[:1000])
    return text


def preprocess_data(text, vocab_idx, seq_len=60, step_num=3):
    x_train = []
    y_train = []

    x_train_onehot = []
    y_train_onehot = []

    for i in range(0, len(text) - seq_len, 3):
        x_train.append(text[i: i+seq_len])
        y_train.append(text[i+seq_len: i+seq_len+1])

    print(f"num of datasets: {len(x_train)}, {len(y_train)}")

    # x_train_onehot = np.zeros((len(x_train), seq_len, len(vocab_idx.keys())))
    # for i in range(len(x_train)):
    #     for j, character in enumerate(x_train[i]):
    #         index = vocab_idx.get(character)
    #         x_train_onehot[i, j, index - 1] = 1
    #
    # y_train_onehot = np.zeros((len(y_train), 1, len(vocab_idx.keys())))
    # for i in range(len(y_train)):
    #     index = vocab_idx.get(y_train[i])
    #     y_train_onehot[i, 0, index - 1] = 1
    #
    # print(f"x_train_onehot.shape: {x_train_onehot.shape}")
    # print(f"y_train_onehot.shape: {y_train_onehot.shape}")
    # return x_train_onehot, y_train_onehot

    x_train_onehot = []
    for i in range(len(x_train)):
        one_hot = np.zeros((len(x_train[i]), len(vocab_idx.keys())))
        for j, character in enumerate(x_train[i]):
            index = vocab_idx.get(character)
            one_hot[j, index - 1] = 1
            x_train_onehot.append(one_hot)

    y_train_onehot = []
    for i in range(len(y_train)):
        one_hot = np.zeros((len(y_train[i]), len(vocab_idx.keys())))
        index = vocab_idx.get(y_train[i])
        one_hot[0, index - 1] = 1
        y_train_onehot.append(one_hot)

    print(f"x_train_onehot.shape: {len(x_train_onehot), x_train_onehot[0].shape}")
    print(f"y_train_onehot.shape: {len(y_train_onehot), y_train_onehot[0].shape}")
    return x_train_onehot, y_train_onehot


def make_vocabulary(text):
    # 1. tokenization
    # 2. Count word frequencies
    vocab = {}
    for i in range(len(text)):
        if vocab.get(text[i]) is not None:
            vocab[text[i]] += 1
        else:
            vocab[text[i]] = 1
    # 3. sort - reverse=True - 降序
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    # 4. map every word to its index
    vocab_idx = {}
    for i in range(len(vocab)):
        vocab_idx[vocab[i][0]] = i + 1
    print(f"count of vocab_idx: {len(vocab_idx)}")
    return vocab_idx


class Vocab(object):
    def __init__(self, text):
        print(text)
        # 1. tokenization
        # 2. Count word frequencies
        vocab = {}
        for i in range(len(text)):
            if vocab.get(text[i]) is not None:
                vocab[text[i]] += 1
            else:
                vocab[text[i]] = 1
        # 3. sort - reverse=True - 降序
        vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        # 4. map every word to its index
        vocab_idx = {}
        for i in range(len(vocab)):
            vocab_idx[vocab[i][0]] = i + 1
        print(f"count of vocab_idx: {len(vocab_idx)}")
        self.vocab_idx = vocab_idx

        one_hot_map = []
        for i in self.vocab_idx.values():
            one_hot = []
            for j in range(i - 1):
                one_hot.append(0)
            one_hot.append(1)
            for k in range(i, len(self.vocab_idx.keys())):
                one_hot.append(0)
            one_hot_map.append(one_hot)

        print(f"one_hot_map.shape: {len(one_hot_map), len(one_hot_map[0])}")
        self.one_hot_map = one_hot_map


import keras.models
from keras import layers

def build_model(seq_len, vocab):
    print(f"input_shape: {seq_len, len(vocab.keys())}")

    model = keras.models.Sequential()
    model.add(layers.LSTM(128, input_shape=(seq_len, len(vocab.keys()))))
    model.add(layers.Dense(len(vocab.keys()), activation='softmax'))
    model.summary()
    optimizer = keras.optimizers.RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

# test - 1
def char_onehot_test():
    samples = ['新华社受权于18日', '全文播发修改后的《中', '华人民共和国立法法》']
    print(f"count of sample: {len(samples)}")
    print(samples)

    str = '新华社受权于18日全文播发修改后的《中华人民共和国立法法》'
    vocab_idx = make_vocabulary(text=str)
    print(len(vocab_idx.keys()))
    print(vocab_idx.keys())

    x_train = np.zeros((len(samples), len(samples[0]) + 1, len(vocab_idx.keys()), ))
    print(x_train.shape)
    for i, sample in enumerate(samples):
        for j, character in enumerate(sample):
            index = vocab_idx.get(character)
            print(sample, character, i, j, index)
            x_train[i, j, index - 1] = 1

    print(x_train.shape)
    print(x_train)


# test - 2
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )

    def forward(self, input):
        hidden = torch.zeros(
            self.num_layers,
            self.batch_size,
            self.hidden_size
        )
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)


# test - 2
def rnn_test():
    # 1.构建数据集
    idx2char = ['e', 'h', 'l', 'o']
    x_data = [1, 0, 2, 2, 3]
    y_data = [3, 1, 2, 3, 2]
    one_hot_lookup = [
        [1, 0, 0, 0],  # 对应one_hot_lookup[0]
        [0, 1, 0, 0],  # 对应one_hot_lookup[1]
        [0, 0, 1, 0],  # 对应one_hot_lookup[2]
        [0, 0, 0, 1]   # 对应one_hot_lookup[3]
    ]
    # 通过字典的查询组成x
    x_one_hot = [one_hot_lookup[x] for x in x_data]
    # [[0, 1, 0, 0],
    # [1, 0, 0, 0],
    # [0, 0, 1, 0],
    # [0, 0, 1, 0],
    # [0, 0, 0, 1]]
    print(x_one_hot)
    inputs = torch.Tensor(x_one_hot)
    print(inputs.shape)  # torch.Size([5, 4])

    input_size = 4
    hidden_size = 4
    batch_size = 1
    num_layers = 1
    seq_len = 5

    inputs = inputs.view(-1, batch_size, input_size)  # [seqlen,batch_size,input_size]
    print(inputs.shape)  # torch.Size([5, 1, 4])

    labels = torch.LongTensor(y_data)
    print(labels.shape)  # torch.Size([5, 1])  [seqlen,1]

    net = Model(input_size, hidden_size, batch_size, num_layers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

    for epoch in range(200):
        loss = 0
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, idx = outputs.max(dim=1)

        idx = idx.data.numpy()
        print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')  # join将预测的字符拼接为一个字符串
        print(',Epoch [%d/15] loss =%.3f' % (epoch + 1, loss.item()))


if __name__ == "__main__":
    char_onehot_test()

    # text = load_lcsts()
    # vocab_idx = make_vocabulary(text[:15])
    # x_train, y_train = preprocess_data(text[:15], vocab_idx, seq_len=10, step_num=5)
    # model = build_model(seq_len=10, vocab=vocab_idx)
    # model.fit(x_train, y_train, batch_size=128, epochs=1)

    # rnn_test()

    # text = load_lcsts()
    # vocab = Vocab(text[:100])



