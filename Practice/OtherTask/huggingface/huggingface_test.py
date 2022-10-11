# -*- coding: utf-8 -*-
# @Time    : 2022/3/13 14:07
# @Author  : Zhang Jiaqi
# @File    : test0_bert_encode.py
# @Description:

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))

