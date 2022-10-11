# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 18:55
# @Author  : Zhang Jiaqi
# @File    : test0_pipeline_usage.py
# @Description:

from transformers import pipeline


# distilbert-base-uncased-finetened-sst-2-english

classifier = pipeline("sentiment-analysis")
print(classifier("We are very happy to show you the Transformers library."))
