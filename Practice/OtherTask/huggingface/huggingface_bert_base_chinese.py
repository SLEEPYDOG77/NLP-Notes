# -*- coding: utf-8 -*-
# @Time    : 2022/3/13 14:49
# @Author  : Zhang Jiaqi
# @File    : test1_bert_base_chinese.py
# @Description: https://www.bilibili.com/video/BV1a44y1H7Jc?p=3&spm_id_from=pageDriver

from datasets import load_dataset
from datasets import load_from_disk
from datasets import load_metric
from transformers import BertTokenizer
from transformers import pipeline

from utils.dict_operate import dict_operate


def tokenizer_test():
    # 加载预训练字典和分词方法
    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name_or_path='bert-base-chinese',
        cache_dir=None,
        force_download=False,
    )

    sents = [
        '选择珠江花园的原因就是方便',
        '笔记本的键盘确实爽',
        '房间太小。其他的都一般。',
        '今天才知道这书还有第6卷，真有点郁闷。',
        '机器背面似乎被撕了张什么标签，残胶还在。',
    ]

    # 批量编码句子
    out = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=[sent for sent in sents],
        add_special_tokens=True,
        truncation=True,
        padding='max_length',
        max_length=15,
        return_tensors=None,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_length=True,
    )

    for index, input_id in enumerate(out['input_ids']):
        print(sents[index])
        print(input_id)

    my_dict = dict_operate(tokenizer=tokenizer)
    vocab = my_dict.get_vocab()
    print("\nLength of Vocabulary: %s\n" % len(vocab))


    for index, token in enumerate(out['input_ids'][0]):
        print(token, list(vocab.keys())[list(vocab.values()).index(token)])


def metric_test():
    pass
    # metric = load_metric('glue', 'mrpc')
    # predictions = [0, 1, 0]
    # references = [0, 1, 1]
    #
    # final_score = metric.compute(predictions=predictions, references=references)
    # print(final_score)


def pipeline_test():
    # sentiment_analysis()
    pass

def data_process():
    pass




if __name__ == "__main__":
    tokenizer_test()
    # load_datasets_from_huggingface()
    # metric_test()
    # pipeline_test()

