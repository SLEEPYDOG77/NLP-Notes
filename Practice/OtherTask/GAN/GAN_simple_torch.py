# -*- coding: utf-8 -*-
# @Time    : 2022/5/16 15:32
# @Author  : Zhang Jiaqi
# @File    : GAN_simple_torch.py
# @Description: https://github.com/hahahappyboy/GAN-Thesis-Retrieval/blob/main/GAN/GAN.py

import cv2
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dis(x)
        return x


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),  # 输入为100维的随机噪声
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.gen(x)
        return x


def load_mnist_data():
    train_dataset = datasets.MNIST(root='data/', train=True,
                                   transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='data/', train=False,
                                   transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)

    images, lables = next(iter(train_loader))
    img = torchvision.utils.make_grid(images, nrow=10)
    img = img.numpy().transpose(1, 2, 0)
    cv2.imshow('img', img)
    cv2.waitKey(0)

    return train_dataset, test_dataset, train_loader, test_loader


def to_img(x):# 将结果的-0.5~0.5变为0~1保存图片
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return


def train(train_loader):
    D = discriminator()
    G = generator()
    if torch.cuda.is_available():  # 放入GPU
        D = D.cuda()
        G = G.cuda()

    # Binary cross entropy loss and optimizer
    criterion = nn.BCELoss()  # BCELoss 因为可以当成是一个分类任务，如果后面不加Sigmod就用BCEWithLogitsLoss
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)  # 优化器
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)  # 优化器

    batch_size = 128
    num_epoch = 100
    z_dimension = 100

    for epoch in range(num_epoch):
        for i, (img, _) in enumerate(train_loader):
            num_img = img.size(0)
            img = img.view(num_img, -1)
            real_img = img.cuda()
            real_label = torch.ones(num_img).reshape(num_img, 1).cuda()  # 希望判别器对real_img输出为1 [128,]
            fake_label = torch.zeros(num_img).reshape(num_img, 1).cuda()  # 希望判别器对fake_img输出为0  [128,]




if __name__ == "__main__":
    train_dataset, test_dataset, train_loader, test_loader = load_mnist_data()
