# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 18:44
# @Author  : Zhang Jiaqi
# @File    : test0_bert_encode.py
# @Description: Huggingface简介及BERT代码浅析 - yangDDD的文章 - 知乎 https://zhuanlan.zhihu.com/p/120315111

import torch
from transformers import BertModel, BertTokenizer

def test():
    """
    读取一个预训练过的BERT模型，来encode我们指定的一个文本
    :return:
    """
    # 模型名称
    model_name = "bert-base-uncased"

    # 读取模型对应的tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # 载入模型
    model = BertModel.from_pretrained(model_name)
    # 输入文本
    input_text = "Here is some text to encode"
    print(input_text)
    # [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
    # ['[CLS]', 'here', 'is', 'some', 'text', 'to', 'en', '##code', '[SEP]']

    # 通过 tokenizer 把文本变成 token_id
    input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    print(input_ids)

    input_ids = torch.tensor([input_ids])
    print(input_ids)

    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
        print(last_hidden_states)


if __name__ == "__main__":
    test()
