# -*- coding: utf-8 -*-
# @Time    : 2022/4/9 10:02
# @Author  : Zhang Jiaqi
# @File    : rnn_poetry.py
# @Description:

import re
import json
import math
import time
import collections
from tqdm import tqdm
import tensorflow as tf
import d2l.tensorflow as d2l
from utils.log_operate import LogOperator

def read_data():
    filepath = '../0data/chinese_poetry_data/train.txt'
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for index, line in enumerate(lines):
        lines[index] = line.replace("\n", "").replace("，", " ").replace("。", " ")
    return lines


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元
    Defined in :numref:`sec_text_preprocessing`"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


def count_corpus(tokens):
    """统计词元的频率
    Defined in :numref:`sec_text_preprocessing`"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def load_corpus(max_tokens=-1):
    lines = read_data()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    print(vocab.token_freqs)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab


class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        """Defined in :numref:`sec_language_model`"""
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


def load_data(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab


def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return tf.random.normal(shape=shape, stddev=0.01, mean=0, dtype=tf.float32)

    # 隐藏层参数
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(tf.zeros(num_hiddens), dtype=tf.float32)
    # 输出层参数
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(tf.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params


def init_rnn_state(batch_size, num_hiddens):
    return (tf.zeros((batch_size, num_hiddens)), )


def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        X = tf.reshape(X, [-1, W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return tf.concat(outputs, axis=0), (H,)


class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward_fn, get_params):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward_fn
        self.trainable_variables = get_params(vocab_size, num_hiddens)

    def __call__(self, X, state):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, self.trainable_variables)

    def begin_state(self, batch_size, *args, **kwargs):
        return self.init_state(batch_size, self.num_hiddens)


def predict(prefix, num_preds, net, vocab):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, dtype=tf.float32)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: tf.reshape(tf.constant([outputs[-1]]), (1, 1)).numpy()

    # 预热期
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])

    # 预测num_preds步
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(grads, theta):
    """裁剪梯度"""
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad


def build_log():
    filepath = 'experiments/rnn_poetry_' + time.strftime("%Y%m%d", time.localtime()) + '/'
    filename = 'rnn_poetry_' + str(int(time.time())) + '.log'
    fullpath = filepath + filename
    return fullpath


def train_epoch(net, train_iter, loss, updater, use_random_iter):
    state, timer = None, d2l.Timer()

    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            y_hat, state = net(X, state)
            y = tf.reshape(tf.transpose(Y), (-1))
            l = loss(y, y_hat)
        params = net.trainable_variables
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))
        # Keras默认返回一个批量中的平均损失
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train(net, train_iter, vocab, lr, num_epochs, strategy, use_random_iter=False, log_opt=None):
    with strategy.scope():
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)
    pred = lambda prefix: predict(prefix, 50, net, vocab)
    device = d2l.try_gpu()._device_name
    # 训练和预测
    for epoch in tqdm(range(num_epochs)):
        ppl, speed = train_epoch(net, train_iter, loss, updater, use_random_iter)
        log = f"epoch {epoch}: {pred('风')}\n困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}"
        print(log)
        log_opt.write_log(log)


def test():
    train_iter, vocab = load_data(batch_size=32, num_steps=35)

    X = tf.reshape(tf.range(10), (2, 5))
    tf.one_hot(tf.transpose(X), 28).shape

    # 定义tensorflow训练策略
    device_name = d2l.try_gpu()._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)

    num_hiddens = 512
    with strategy.scope():
        net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn,
                              get_params)
    state = net.begin_state(X.shape[0])
    Y, new_state = net(X, state)
    print(Y.shape, len(new_state), new_state[0].shape)

    pred = predict('昨天', 10, net, vocab)
    print(pred)


if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data(batch_size, num_steps, use_random_iter=True)

    device_name = d2l.try_gpu()._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)

    num_hiddens = 512
    with strategy.scope():
        net = RNNModelScratch(len(vocab), num_hiddens, init_rnn_state, rnn, get_params)

    num_epochs, lr = 500, 1

    logpath = build_log()
    log_opt = LogOperator(logpath)

    params = {
        "model": "RNN",
        "dataset": "chinese-poetry",
        "batch_size": batch_size,
        "num_steps": num_steps,
        "num_hiddens": num_hiddens,
        "num_epochs": num_epochs,
        "lr": lr
    }
    log_opt.write_model(params)
    train(net, train_iter, vocab, lr, num_epochs, strategy, log_opt=log_opt)


