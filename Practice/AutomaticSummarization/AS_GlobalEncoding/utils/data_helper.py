# -*- coding: utf-8 -*-
# @Time    : 2022/4/3 10:41
# @Author  : Zhang Jiaqi
# @File    : data_helper.py
# @Description:
import torch
import linecache
import torch.utils.data as torch_data

class BiDataset(torch_data.Dataset):

    def __init__(self, infos, indexes=None, char=False):

        self.srcF = infos['srcF']
        self.tgtF = infos['tgtF']
        self.original_srcF = infos['original_srcF']
        self.original_tgtF = infos['original_tgtF']
        self.length = infos['length']
        self.infos = infos
        self.char = char
        if indexes is None:
            self.indexes = list(range(self.length))
        else:
            self.indexes = indexes

    def __getitem__(self, index):
        index = self.indexes[index]
        src = list(map(int, linecache.getline(self.srcF, index+1).strip().split()))
        tgt = list(map(int, linecache.getline(self.tgtF, index+1).strip().split()))
        original_src = linecache.getline(self.original_srcF, index+1).strip().split()
        original_tgt = linecache.getline(self.original_tgtF, index+1).strip().split() if not self.char else \
                       list(linecache.getline(self.original_tgtF, index + 1).strip())

        return src, tgt, original_src, original_tgt

    def __len__(self):
        return len(self.indexes)

def padding(data):
    src, tgt, original_src, original_tgt = zip(*data)

    src_len = [len(s) for s in src]
    src_pad = torch.zeros(len(src), max(src_len)).long()
    for i, s in enumerate(src):
        end = src_len[i]
        src_pad[i, :end] = torch.LongTensor(s[end-1::-1])

    tgt_len = [len(s) for s in tgt]
    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
    for i, s in enumerate(tgt):
        end = tgt_len[i]
        tgt_pad[i, :end] = torch.LongTensor(s)[:end]

    return src_pad, tgt_pad, \
           torch.LongTensor(src_len), torch.LongTensor(tgt_len), \
           original_src, original_tgt

